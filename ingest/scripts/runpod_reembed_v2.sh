#!/usr/bin/env bash
# On-pod I6 re-embed: exported payload rows → v2-header embed → pod-local Qdrant.
# Same clean-venv + static-Qdrant setup as runpod_embed_delta.sh; rows are expected
# unpacked at $WORK/rows (delivered by the orchestrator inside the encrypted payload).
# SHARDS>1 launches one reembed_v2.py process per GPU. Idempotent/resumable (per-shard
# checkpoints inside $WORK/rows).
set -uo pipefail

WORK="${WORK:-/workspace}"
OUT="$WORK/out"
COLLECTION="${COLLECTION_NAME:-georgian_legal_v2}"
QDRANT_VER="${QDRANT_VER:-v1.18.2}"
N="${SHARDS:-1}"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/embed.log") 2>&1
echo "[$(date -u +%FT%TZ)] reembed_v2 start  shards=$N  gpus=$(nvidia-smi -L 2>/dev/null | wc -l)"
nvidia-smi -L 2>/dev/null

test -d "$WORK/ingest" || { echo "FATAL: $WORK/ingest not present (payload not unpacked)"; exit 1; }
test -d "$WORK/rows" || { echo "FATAL: $WORK/rows not present (export not delivered)"; exit 1; }

# --- venv (clean, local-matched stack) ------------------------------------------
PY="$WORK/venv/bin/python"
if [ ! -x "$PY" ]; then python -m venv "$WORK/venv"; fi
set -e
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q torch --index-url https://download.pytorch.org/whl/cu124
"$PY" -m pip install -q "FlagEmbedding==1.4.0" "transformers==5.12.1" "tokenizers==0.22.2" \
  "accelerate==1.14.0" "datasets==5.0.0" "sentencepiece==0.2.1" "safetensors==0.8.0" \
  "qdrant-client>=1.12" "python-dotenv>=1.0" "tqdm>=4.66" "rich>=13"
"$PY" -c "import torch; from FlagEmbedding import BGEM3FlagModel; assert torch.cuda.is_available(); \
  print('deps OK: torch', torch.__version__, 'gpus', torch.cuda.device_count())"

# --- Qdrant static binary ---------------------------------------------------------
if ! curl -sf http://127.0.0.1:6333/ >/dev/null 2>&1; then
  cd "$WORK"
  curl -sSL -o qdrant.tar.gz \
    "https://github.com/qdrant/qdrant/releases/download/${QDRANT_VER}/qdrant-x86_64-unknown-linux-musl.tar.gz"
  tar xzf qdrant.tar.gz --no-same-owner --no-same-permissions
  ( QDRANT__STORAGE__STORAGE_PATH="$WORK/qdrant_storage" \
    QDRANT__STORAGE__SNAPSHOTS_PATH="$WORK/qdrant_snapshots" \
    QDRANT__SERVICE__MAX_REQUEST_SIZE_MB=1024 \
    ./qdrant >"$OUT/qdrant.log" 2>&1 & )
  for i in $(seq 1 60); do curl -sf http://127.0.0.1:6333/ >/dev/null 2>&1 && break; sleep 1; done
fi

# --- shard-parallel v2 embed ------------------------------------------------------
cd "$WORK/ingest"
export QDRANT_URL="http://127.0.0.1:6333" COLLECTION_NAME="$COLLECTION"
export EMBED_DEVICE="cuda" EMBED_USE_FP16="true" EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-256}"

# shard 0 creates the collection before the others start upserting
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/reembed_v2.py --rows "$WORK/rows" \
  --collection "$COLLECTION" --shard 0 --num-shards "$N" --header v2 \
  > "$OUT/shard0.log" 2>&1 &
PIDS=($!)
sleep 20
for i in $(seq 1 $((N - 1))); do
  CUDA_VISIBLE_DEVICES=$i "$PY" scripts/reembed_v2.py --rows "$WORK/rows" \
    --collection "$COLLECTION" --shard "$i" --num-shards "$N" --header v2 \
    > "$OUT/shard$i.log" 2>&1 &
  PIDS+=($!)
done
FAIL=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=1; done
tail -n 2 "$OUT"/shard*.log
[ "$FAIL" -eq 0 ] || { echo "FATAL: a shard failed"; exit 1; }

# --- verify counts vs the export manifest, then snapshot --------------------------
EXPECTED=$("$PY" -c "import json;print(json.load(open('$WORK/rows/manifest.json'))['rows'])")
for i in $(seq 1 120); do  # wait for wait=False upserts to settle
  POINTS=$(curl -sf "http://127.0.0.1:6333/collections/${COLLECTION}" \
    | "$PY" -c "import sys,json; print(json.load(sys.stdin)['result']['points_count'])")
  [ "$POINTS" = "$EXPECTED" ] && break
  sleep 5
done
echo "[$(date -u +%FT%TZ)] points=$POINTS expected=$EXPECTED"
[ "$POINTS" = "$EXPECTED" ] || { echo "FATAL: point count mismatch"; exit 1; }

SNAP=$(curl -sf -X POST "http://127.0.0.1:6333/collections/${COLLECTION}/snapshots" \
  | "$PY" -c "import sys,json; print(json.load(sys.stdin)['result']['name'])")
cp "$WORK/qdrant_snapshots/${COLLECTION}/${SNAP}" "$OUT/${COLLECTION}.snapshot"
echo "[$(date -u +%FT%TZ)] reembed_v2 DONE  points=$POINTS  snapshot=$SNAP"
echo "$POINTS" > "$OUT/DONE"
