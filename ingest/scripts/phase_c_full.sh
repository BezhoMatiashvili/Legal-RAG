#!/usr/bin/env bash
# Phase C full sweep: measure the retrieval stack over the green 2.45M-chunk index against
# the 103-pair golden set. Every run appends a row (metrics + bootstrap CIs + per-stage
# latency + per-type/lang) to eval/experiments.jsonl via --log. Neural modes run on CPU.
#
# Usage:  bash scripts/phase_c_full.sh {tier1|tier2|docs|all}
#   tier1 = gate-critical (core modes + routing + rerank-depth) — run first, cheap-to-measure
#   tier2 = exhaustive extras (diversity + Qdrant/CPU tuning grid)
#   docs  = optional doc-level relevance pass over the core modes
#
# Notes: bm25 uses the prebuilt full-corpus index (eval/.bm25_full/). rerank@80 == plain
# rerank (default rerank_candidates=80), so the depth ablation only adds 10/30/50.
set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"
PY=.venv/bin/python
NOISE='Fetching|Loading checkpoint|it/s|it\]|httpx|resolve|colbert|DeprecationWarning|Swig|^[[:space:]]*$'

run() {  # run <label> <evaluate-args...>
  local label="$1"; shift
  # Only rerank modes need the cross-encoder. Loading it for other modes wastes ~2.3 GB and,
  # on this RAM-tight box (30 GB, Qdrant resident ~21 GB), evicts Qdrant's hot sparse index to
  # swap → ~275 s/query thrash. Disabling it for non-rerank modes changes neither results nor
  # config_hash (the eval hash ignores rerank_enabled; the reranker is unused there).
  local rr=false
  case " $* " in *" --mode rerank "*) rr=true ;; esac
  local t0; t0=$(date +%s)
  echo "[$(date +%T)] >>> $label :: RERANK_ENABLED=$rr :: $*"
  # timeout guards against any residual thrash: cap a single invocation at 60 min.
  RERANK_ENABLED="$rr" timeout 3600 $PY -m eval.evaluate "$@" 2>&1 | grep -avE "$NOISE"
  local t1; t1=$(date +%s)
  echo "[$(date +%T)] <<< $label DONE in $((t1 - t0))s"
  echo "--------------------------------------------------------------------------------"
}

tier1() {
  echo "===== TIER 1 : core modes + routing + rerank depth ($(date)) ====="
  run bm25   --backend qdrant --mode bm25   --relevance chunk --log
  run dense  --backend qdrant --mode dense  --relevance chunk --log
  run sparse --backend qdrant --mode sparse --relevance chunk --log
  run rerank80 --backend qdrant --mode rerank --relevance chunk --log
  # compare logs hybrid + routed rows AND prints the paired A/B (the load-bearing routing test)
  run cmp_hybrid_routed --backend qdrant --compare hybrid routed --relevance chunk --log
  run rerank10 --backend qdrant --mode rerank --rerank-candidates 10 --relevance chunk --log
  run rerank30 --backend qdrant --mode rerank --rerank-candidates 30 --relevance chunk --log
  run rerank50 --backend qdrant --mode rerank --rerank-candidates 50 --relevance chunk --log
  echo "===== TIER 1 COMPLETE ($(date)) ====="
}

diversity() {
  # Rerank-bearing → offloaded to GPU (export RERANK_REMOTE_URL=http://localhost:8900 first);
  # on CPU each is ~a rerank run. STANDALONE (not --ab) so each has a distinct config_hash and
  # never collides with the CPU rerank@80 baseline row (which carries the production latency).
  echo "===== DIVERSITY grid : ${RERANK_REMOTE_URL:+GPU }max-per-doc + MMR ($(date)) ====="
  for m in 2 3 5; do
    run "div_mpd${m}" --backend qdrant --mode rerank --max-per-doc "$m" --relevance chunk --log
  done
  for l in 0.3 0.5 0.7; do
    run "div_mmr${l}" --backend qdrant --mode rerank --mmr-lambda "$l" --relevance chunk --log
  done
  echo "===== DIVERSITY COMPLETE ($(date)) ====="
}

tuning() {
  # All non-rerank (hybrid/dense) → fast on CPU, no reranker, no thrash.
  echo "===== TUNING grid : fusion + prefetch + ef/rescore + sparse-weight ($(date)) ====="
  run fusion_dbsf_ab --backend qdrant --mode hybrid --fusion dbsf --ab --relevance chunk --log
  for pf in 50 100 200 400; do
    run "prefetch_${pf}" --backend qdrant --mode hybrid --prefetch-limit "$pf" --relevance chunk --log
  done
  for ef in 64 128 256; do
    run "ef${ef}_rescore_on"  --backend qdrant --mode dense --hnsw-ef "$ef" --rescore on  --relevance chunk --log
    run "ef${ef}_rescore_off" --backend qdrant --mode dense --hnsw-ef "$ef" --rescore off --relevance chunk --log
  done
  for w in 0.0 0.2 0.5 0.8 1.0; do
    run "sparseweight_${w}" --backend qdrant --mode hybrid --sparse-weight "$w" --relevance chunk --log
  done
  echo "===== TUNING COMPLETE ($(date)) ====="
}

gpu_rerank() {
  # Full rerank grid on the GPU pod (retrieval stays local). Requires the SSH-tunnel endpoint.
  # Logged to a SEPARATE file: GPU rows give rerank/diversity QUALITY; the row latency is
  # local-retrieval + tunnel round-trip, NOT CPU serving latency — use rerank_latency_probe.py
  # for the CPU rerank latency the report reports.
  : "${RERANK_REMOTE_URL:?export RERANK_REMOTE_URL=http://localhost:8900 (GPU pod tunnel) first}"
  local GLOG=eval/experiments_gpu.jsonl
  echo "===== GPU RERANK grid via $RERANK_REMOTE_URL ($(date)) ====="
  for d in 10 30 50 80; do
    run "gpu_rerank${d}" --backend qdrant --mode rerank --rerank-candidates "$d" \
      --relevance chunk --log --log-path "$GLOG"
  done
  for m in 2 3 5; do
    run "gpu_div_mpd${m}" --backend qdrant --mode rerank --max-per-doc "$m" \
      --relevance chunk --log --log-path "$GLOG"
  done
  for l in 0.3 0.5 0.7; do
    run "gpu_div_mmr${l}" --backend qdrant --mode rerank --mmr-lambda "$l" \
      --relevance chunk --log --log-path "$GLOG"
  done
  echo "===== GPU RERANK COMPLETE ($(date)) ====="
}

docs() {
  echo "===== DOC-LEVEL relevance pass (core modes) ($(date)) ====="
  for m in bm25 dense sparse hybrid rerank routed; do
    run "doc_${m}" --backend qdrant --mode "$m" --relevance doc --log
  done
  echo "===== DOC-LEVEL COMPLETE ($(date)) ====="
}

case "${1:-all}" in
  tier1)      tier1 ;;
  gpu_rerank) gpu_rerank ;; # depth + diversity via the GPU pod (needs RERANK_REMOTE_URL)
  diversity)  diversity ;;  # export RERANK_REMOTE_URL first to run on the GPU pod
  tuning)     tuning ;;
  tier2)     diversity; tuning ;;
  docs)      docs ;;
  all)       tier1; diversity; tuning ;;
  *) echo "usage: $0 {tier1|diversity|tuning|tier2|docs|all}"; exit 2 ;;
esac
