#!/usr/bin/env bash
# On-pod MULTI-SOURCE delta embed: staged <source>.jsonl files → georgian_legal_delta.
# Same clean-venv + static-Qdrant setup as runpod_embed_delta.sh; the staged per-source
# missing items are expected unpacked at $WORK/delta_items/<source>.jsonl.
set -uo pipefail

WORK="${WORK:-/workspace}"
OUT="$WORK/out"
COLLECTION="${COLLECTION_NAME:-georgian_legal_delta}"
QDRANT_VER="${QDRANT_VER:-v1.18.2}"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/embed.log") 2>&1
echo "[$(date -u +%FT%TZ)] multi-source delta start  gpus=$(nvidia-smi -L 2>/dev/null | wc -l)"
nvidia-smi -L 2>/dev/null

test -d "$WORK/ingest" || { echo "FATAL: $WORK/ingest not present"; exit 1; }
test -d "$WORK/delta_items" || { echo "FATAL: $WORK/delta_items not present"; exit 1; }

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

cd "$WORK/ingest"
export QDRANT_URL="http://127.0.0.1:6333" COLLECTION_NAME="$COLLECTION"
export EMBED_DEVICE="cuda" EMBED_USE_FP16="true" EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-256}"
# v1 headers (EMBED_HEADER_V2 unset) — these merge into the live v1 collection.
"$PY" scripts/embed_delta.py --items-dir "$WORK/delta_items" \
  --collection "$COLLECTION" --batch-size "${EMBED_BATCH_SIZE:-256}"

SNAP=$(curl -sf -X POST "http://127.0.0.1:6333/collections/${COLLECTION}/snapshots" \
  | "$PY" -c "import sys,json; print(json.load(sys.stdin)['result']['name'])")
cp "$WORK/qdrant_snapshots/${COLLECTION}/${SNAP}" "$OUT/${COLLECTION}.snapshot"
POINTS=$(curl -sf "http://127.0.0.1:6333/collections/${COLLECTION}" \
  | "$PY" -c "import sys,json; print(json.load(sys.stdin)['result']['points_count'])")
echo "[$(date -u +%FT%TZ)] multi-source delta DONE  points=$POINTS  snapshot=$SNAP"
echo "$POINTS" > "$OUT/DONE"
