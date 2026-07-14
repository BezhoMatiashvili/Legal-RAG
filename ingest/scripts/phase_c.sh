#!/usr/bin/env bash
# Phase C baseline: run the neural retrieval modes on the golden set, each logged to
# experiments.jsonl with metrics + bootstrap CIs. Skips bm25 (impractical at 2.45M-chunk
# scale in the in-memory harness — handled separately). Runs unattended.
set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")" || exit 1
PY=.venv/bin/python
CLEAN='grep -vE Fetching|Loading|it/s|Warning|httpx|resolve|colbert'
log() { echo "[$(date +%T)] $*"; }

log "PHASE C START  (green index, 103-pair golden set)"
for m in dense sparse hybrid rerank; do
  log "=== mode=$m ==="
  $PY -m eval.evaluate --backend qdrant --mode "$m" --relevance chunk --log \
    2>&1 | grep -vE "Fetching|Loading|it/s|Warning|httpx|resolve|colbert|^\s*$"
  log "--- mode=$m done ---"
done

log "=== paired A/B: hybrid vs routed (cross-lingual routing) ==="
$PY -m eval.evaluate --backend qdrant --compare hybrid routed --relevance chunk --log \
  2>&1 | grep -vE "Fetching|Loading|it/s|Warning|httpx|resolve|colbert|^\s*$"

log "PHASE C BASELINE DONE"
