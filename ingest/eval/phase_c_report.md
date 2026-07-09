# Part 3 Phase C/D — Retrieval evaluation report

_Georgian legal RAG · first real full-corpus baseline + hardening measurements · 2026-07-09_

## Executive summary

Measured the whole retrieval stack over the **permanent full-corpus index** (Qdrant
`georgian_legal`, 2,453,915 chunks, G2-verified) against the 103-pair span-anchored golden set,
each metric with 95 % bootstrap CIs, with a **mandatory full-corpus BM25 floor**.

1. **The neural stack beats the BM25 floor on every metric, by large point-estimate margins.**
   Best config (hybrid + rerank@80): nDCG@10 **0.289 vs BM25 0.192 (+50 %)**, R@10 0.388 vs 0.243,
   MRR@10 0.270 vs 0.176. **Significance caveat:** these are point estimates; the 95 % per-mode CIs
   overlap at n=103 and **no paired rerank-vs-baseline test was run** (only routing got one). A
   paired permutation test is the recommended follow-up to certify the rerank win (§10).
2. **Reranking is the largest lever by point estimate.** Hybrid→+rerank lifts nDCG 0.182→0.289 and
   MRR 0.146→0.270 — the cross-encoder's precision/ordering gain, exactly as designed (same
   significance caveat as #1).
3. **Cross-lingual is recovered only by the neural stack.** On the 22 EN→KA pairs, BM25 and sparse
   are **structurally 0.000** (lexical cannot bridge languages — a qualitative certainty, not a
   marginal effect); the neural stack reaches **rerank@80 R@10 0.273 (6/22 hits)**. This
   rerank-vs-lexical gap is the whole justification for the neural stack. _The depth ordering within
   rerank (3→4→5→6 of 22 hits at depth 10→30→50→80) trends up but is on n=22 with no CIs — treat
   "deeper helps cross-lingual" as suggestive, not established._
4. **Language routing is a statistical TIE** (routed vs hybrid, paired permutation test: all four
   metrics TIE, p ≥ 0.11). Dropping sparse for EN neither helps nor hurts here.
5. **Diversity: don't enable it.** Per-doc cap = tie (identical to displayed precision; nDCG/MRR
   differ <0.001); **MMR is significantly harmful** (nDCG 0.289 → 0.074, CI [0.037,0.117] disjoint
   from the base [0.212,0.367]) — legal queries want the one on-point chunk, not "diverse" ones.
6. **Recommended interim production config: `hybrid + rerank@50`, no diversity** —
   `retrieval_fingerprint = 81c807b279399098` (see §6). Depth is a documented latency/quality knob.

**Binding caveat:** CPU rerank latency is the real constraint — now **measured** (reranker-only,
OMP_NUM_THREADS=8, `scripts/rerank_latency_probe.py`): p50 **16.2 s @10 · 45.4 s @30 · 66.9 s @50 ·
114.4 s @80** per query. Worse than the earlier derived estimates (~25 s @50) — this box is a
hybrid P/E-core laptop CPU that power-scales under sustained load. CPU rerank is definitively not
interactive; an int8/ONNX or GPU-accelerated reranker (the GPU path built this session) is
mandatory before deep rerank ships interactively.

---

## 1. What this measures

- **Index:** local Qdrant `georgian_legal` — 2,453,915 chunks (dense BGE-M3 1024-d `on_disk` +
  int8 `always_ram`; BGE-M3 learned-sparse in-RAM), G2-verified (CPU↔GPU cosine 0.999999),
  persisted on the docker bind-mount. Per-source: matsne 1,855,148 · ecd 383,452 · napr 141,427 ·
  constcourt 71,292 · tas 2,400 · tbappeal 196.
- **Eval set:** `golden_set_v1` — 103 span-anchored query→chunk pairs / 51 holdout docs.
  Types: natural_question 32, cross_lingual 22 (EN→KA), keyword 21, legal_citation 17, paraphrase 11;
  ka 81 / en 22 (the 22 EN == the cross_lingual slice). Fail-loud guards pass (re-grounding 103/0,
  holdout, span-coverage 103/0).
- **Metrics:** Recall@5/@10 (hit-rate), nDCG@10 (graded), MRR@10 — chunk-level, with per-type /
  per-language breakdowns and per-stage CPU latency. **Stats:** percentile bootstrap 95 % CIs +
  two-sided paired permutation test (adopt only if p<0.05 **and** the paired-difference CI excludes 0).
  All runs appended to `eval/experiments.jsonl` (+ GPU-offloaded rerank to `eval/experiments_gpu.jsonl`),
  each with `config_hash`, eval-set version/hash and a self-describing `knobs` field.

## 2. Phase A embed (recap)

Full corpus embedded on RunPod (4× RTX 4090, CUDA FP16), 2,453,915 chunks. **G2 vector-identity
PASSED** (cosine 0.999999) — GPU vectors live in the CPU reference space, so CPU-served retrieval is
trustworthy. **Pod-hours/cost:** the embed *compute* was ~45 min on the 4× 4090 (~3 GPU-hours at
$2.76/hr); total spend **~$10.70 of $15**, but the bulk of that was pod wall-clock during the flaky
**24 GB snapshot pull** (the pod stayed up hours over a slow home link) + a false-start single-GPU
attempt — not the ~45-min embed itself. Exact pod-hours weren't cleanly logged; the $10.70 is the
authoritative figure. The 24 GB snapshot is backed up (`~/gpu_embed_work/out_multi/`, md5 378985d4…)
— never needs re-embedding unless model/chunking change.

## 3. Compute — CPU serving target, and where GPU was used

Serving runs on CPU. The **full eval** stays on CPU: putting it on a pod would need the 2.45M-chunk
index there (24 GB upload ≈ 13+ h, or a full re-embed ≈ the whole remaining budget), and latency
**must** be CPU (the serving target). BM25 is CPU lexical.

**Reranking is the exception, and GPU genuinely helped there.** Unlike the full eval, the
cross-encoder only needs the small *(query, candidate-text)* pairs (~8 MB/run), not the index — so
the rerank-depth ablation + diversity quality were offloaded to a **RunPod RTX 4090** over an
encrypted SSH tunnel (retrieval + metrics stayed local; a drop-in `RemoteBGEReranker` with identical
model/tokenizer/sigmoid). This turned an ~11 h CPU rerank grid into ~40 min for **~$1.20**; the pod
was auto-terminated (try/finally + atexit + signals + watchdog). CPU rerank **latency** was measured
separately (§3 note in the tables) since GPU latency would mislead the serving recommendation.

## 4. BM25 baseline — method (mandatory floor)

BM25 is the neural-stack sanity floor over the **same chunks**. The reference `eval/bm25.BM25Index`
(Okapi/Lucene, k1=1.5, b=0.75, Unicode `\w+` casefold, non-negative Lucene IDF) is intractable at
2.45M chunks (~17 GB), so `eval/bm25_full.FullCorpusBM25` builds the **identical** index as a compact
CSR term→doc weight matrix in two streaming passes over Qdrant (peak ~3 GB, one-time, cached to
`eval/.bm25_full/`) and queries it via mmap in ms — a true **full-corpus** floor. Scoring parity with
the reference is unit-asserted (`tests/test_bm25_full.py`: exact tie-breaking, empty-corpus, k≤0,
save/load, cache-collection guard; the module was also adversarially reviewed). Index: N=2,453,915,
V=4,213,139, nnz=207,680,637, avgdl=144.2.

## 5. Measurement notes (CPU / RAM)

30 GB box; Qdrant resident ~10 GB RSS (+ page cache for on-disk vectors). Two findings shaped the run:
- **Reranker + RAM:** the harness loaded the reranker for *every* mode, evicting Qdrant's hot sparse
  index to swap → a `--mode sparse` run thrashed to 7.9 h. Fix: load the reranker only for `rerank`
  modes (no change to results or `config_hash`) → sparse back to 76 s. Even so, CPU `rerank@80`
  (both models resident) ran 70 min and OOM-crashed — the reason reranking was moved to GPU.
- **The reranker was subsequently optimized** (physical-core thread pinning + length-bucketed batches,
  `ingest/rerank.py`) — scores unchanged. Per-depth CPU latency has since been **measured directly**
  (`scripts/rerank_latency_probe.py`, 2026-07-09): p50 16.2/45.4/66.9/114.4 s per query at depth
  10/30/50/80. These supersede the old derived numbers (which turned out to be underestimates —
  the box's hybrid P/E-core CPU clocks down under sustained all-core load).

---

## 6. Results

_Full 2.45M-chunk index, 103-query golden set, chunk-level. mean [95 % bootstrap CI]. Companion:
`eval/phase_c_report_tables.md` (regenerable via `scripts/build_phase_c_report.py`)._

### Modes — BM25 floor → dense/sparse/hybrid → +rerank → routed

| config | R@10 | nDCG@10 | MRR@10 | lat p50/p95 (ms) |
|---|---|---|---|---|
| bm25 (floor) | 0.243 [0.165, 0.330] | 0.192 [0.124, 0.267] | 0.176 [0.111, 0.248] | 70 / 243 |
| dense | 0.214 [0.136, 0.291] | 0.106 [0.064, 0.153] | 0.080 [0.045, 0.121] | 323 / 432 |
| sparse | 0.282 [0.194, 0.369] | 0.182 [0.121, 0.248] | 0.160 [0.102, 0.223] | 307 / 405 |
| hybrid | 0.330 [0.243, 0.427] | 0.182 [0.126, 0.244] | 0.146 [0.094, 0.203] | 474 / 642 |
| routed | 0.330 [0.243, 0.427] | 0.187 [0.129, 0.249] | 0.151 [0.098, 0.208] | 440 / 564 |
| **+rerank@80** | **0.388 [0.301, 0.485]** | **0.289 [0.212, 0.367]** | **0.270 [0.195, 0.348]** | 114354 / 116142 (CPU, measured) |

### Cross-lingual slice (22 EN→KA pairs)

| config | R@10 | nDCG@10 | MRR@10 |
|---|---|---|---|
| bm25 / sparse | 0.000 | 0.000 | 0.000 |
| dense / routed | 0.136 | 0.062 | 0.047 |
| hybrid | 0.136 | 0.047 | 0.027 |
| **rerank@80** | **0.273** | **0.184** | **0.156** |

### Rerank-depth ablation (quality: GPU fp32; latency: CPU, measured 2026-07-09)

| depth | R@10 | nDCG@10 | cross-lingual R@10 | lat p50 (ms, measured) |
|---|---|---|---|---|
| rerank@10 | 0.330 [0.243, 0.427] | 0.249 [0.174, 0.326] | 0.136 | 16197 |
| rerank@30 | 0.340 [0.252, 0.437] | 0.262 [0.186, 0.341] | 0.182 | 45429 |
| rerank@50 | 0.359 [0.272, 0.456] | 0.270 [0.193, 0.349] | 0.227 | 66856 |
| rerank@80 | 0.388 [0.301, 0.485] | 0.289 [0.212, 0.367] | 0.273 | 114354 |

Quality rises monotonically with depth on point estimates, but the per-depth CIs overlap (depth
differences are not individually significant at n=103). Cross-lingual R@10 trends up with depth
(3→4→5→6 of 22 hits at depth 10→30→50→80) — **suggestive but uncertified** (n=22, no CIs). So depth
is a latency/quality knob whose cross-lingual payoff is plausible but not statistically established;
pick it against a latency budget, not a significance claim.

### Diversity vs no-diversity rerank@80

| config | R@10 | nDCG@10 [95% CI] | lat p50 (ms) | verdict |
|---|---|---|---|---|
| rerank@80 (base) | 0.388 | 0.289 [0.212, 0.367] | 114354 (CPU, measured) | — |
| max-per-doc=2 | 0.388 | 0.289 [0.212, 0.368] | +cap (negligible) | **tie** (CIs identical) |
| mmr λ=0.5 | 0.136 | 0.074 [0.037, 0.117] | +MMR (dense-vec pass) | **significantly harmful** (CI disjoint from base) |

### Routing — paired significance (hybrid vs routed)

| metric | Δ (routed−hybrid) | 95 % CI | p | verdict |
|---|---|---|---|---|
| R@5 | +0.019 | [0.000, 0.049] | 0.50 | TIE |
| R@10 | 0.000 | [0.000, 0.000] | 1.00 | TIE |
| nDCG@10 | +0.004 | [0.000, 0.009] | 0.11 | TIE |
| MRR@10 | +0.005 | [−0.000, 0.012] | 0.11 | TIE |

## 7. Recommended interim production config

**`hybrid (dense+sparse RRF) + cross-encoder rerank@50, no diversity`** —
`retrieval_fingerprint = 81c807b279399098` (BGE-M3 / BGE-reranker-v2-m3, rerank_candidates=50,
rerank_min_score=0.3, chunk 512/80/64, collection `georgian_legal`).

Rationale:
- **Rerank is non-negotiable** — it is the only thing that lifts nDCG/MRR substantially and the only
  thing that handles cross-lingual (0.047 → 0.184 nDCG on the EN slice).
- **Depth 50** balances the curve: nDCG 0.270 (93 % of @80) and cross-lingual R@10 0.227 (83 % of
  @80) at ~0.63× the rerank cost. Depth is a documented knob — `rerank@30`
  (`5844570078198a3d`, faster) or `rerank@80` (`06a64f548fcb4d59`, best quality + cross-lingual) are
  drop-in alternatives; adopt only against a latency budget.
- **Routing optional** (TIE) — keep hybrid for simplicity; `routed` is an equal-quality alternative
  that drops the sparse branch for EN.
- **Diversity off** (cap neutral, MMR harmful).

**Serving caveat (load-bearing):** CPU rerank latency (**measured 66.9 s/q @50, 114.4 s/q @80**)
is impractical for interactive use. The reranker is the bottleneck — recommend an **int8/ONNX CPU
reranker or the GPU rerank path** (built this session) before this ships interactively.
Retrieval-only (hybrid, ~0.5 s/q) is the graceful-degradation fallback if the reranker is
unavailable.

## 8. Qdrant recall-setting tuning grid (completed 2026-07-09, green index)

All 16 cells ran clean on the green post-backfill index (2,454,410 points; the earlier attempt
had died on "fill query context" timeouts from the concurrent `backfill_consolidation.py`
writer). Full tables in §5–§8 of `phase_c_report_tables.md`. **Every knob is secondary — the
recommendation is unchanged:**

- **Fusion RRF vs DBSF:** paired A/B = **TIE on every metric** (e.g. nDCG@10 Δ=-0.004,
  p=0.78); DBSF's point estimates are slightly worse (R@10 0.291 vs 0.330). Keep **RRF**.
- **Prefetch depth 50→400:** R@10 flat at 0.330, nDCG@10 0.182→0.184 (noise). Keep default.
- **HNSW ef 64/128/256 × int8 rescore (dense):** ef=256 nudges nDCG@10 0.112→0.118 (CIs overlap
  heavily); rescore on/off is a wash. Dense alone is far below hybrid, so no prod impact.
- **Sparse-weight (hybrid):** overall nDCG@10 peaks at w=0.8 (0.191 vs 0.179 base, CIs overlap)
  **but the cross-lingual slice collapses monotonically with sparse weight** (EN R@10:
  0.182 @ w=0.0 → 0.045 @ w=0.8 → 0.000 @ w=1.0) — direct confirmation that lexical/sparse
  matching carries zero cross-lingual signal. The small overall gain is not worth killing the
  EN slice; keep the default RRF hybrid.

## 9. Sample daily-ingestion report

Generated via the real `write_ingest_report()` path → `.state/reports/ingest-20260709.json`
(`kind:"sample"`; the corpus is static, so a live watch shows all-unchanged — this illustrative
sample populates the delta/schema-drift fields). Totals: docs 184,575 · added 15 · updated 32 ·
unchanged 184,528 · chunks 654 · schema_drift {tas: appeal_outcome ×2}. Per-source rows carry
docs/added/updated/unchanged/skipped/chunks + schema_drift + updated_at.

## 10. Caveats & follow-ups

- ~~**Rerank latency is derived**~~ **RESOLVED (2026-07-09):** measured per-depth with the
  length-bucketed reranker (`scripts/rerank_latency_probe.py`, OMP_NUM_THREADS=8, →
  `eval/rerank_xval.json`): p50 16.2/45.4/66.9/114.4 s per query @ depth 10/30/50/80. The old
  derived numbers were ~2.8× optimistic — this hybrid P/E-core CPU throttles under sustained
  all-core load (observed at 45% clock scaling). Conclusion unchanged but stronger: CPU rerank
  is not interactive at any useful depth.
- ~~**Rerank-vs-hybrid significance**~~ **RESOLVED (2026-07-09):** a paired permutation test
  (`--compare hybrid rerank`, rerank scored on a GPU pod, retrieval local) **certifies the rerank
  win**: nDCG@10 Δ=+0.113 [+0.065,+0.165] p=0.0001 → ADOPT; MRR@10 Δ=+0.131 [+0.078,+0.190]
  p=0.0001 → ADOPT; R@5 Δ=+0.117 [+0.039,+0.194] p=0.0072 → ADOPT; R@10 Δ=+0.068 [+0.000,+0.136]
  p=0.095 → TIE. Rows in `eval/experiments_gpu.jsonl`; paired block in
  `~/gpu_embed_work/significance_compare.log`.
- **MMR crater** (0.289→0.074) is directionally expected but large; worth confirming it's a genuine
  effect vs. an `diversify()` MMR-implementation artifact before drawing library-level conclusions.
