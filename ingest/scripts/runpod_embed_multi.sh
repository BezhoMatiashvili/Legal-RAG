#!/usr/bin/env bash
# Multi-GPU on-pod embed: N sharded processes (one per GPU) → one local Qdrant. Same clean-venv
# setup as runpod_embed.sh; the corpus is expected already unpacked at $WORK/ingest (delivered
# pod-to-pod), so no decrypt step. Idempotent/resumable: each shard has its own checkpoint.
set -uo pipefail

WORK="${WORK:-/workspace}"
OUT="$WORK/out"
COLLECTION="${COLLECTION_NAME:-georgian_legal}"
QDRANT_VER="${QDRANT_VER:-v1.18.2}"
N="${SHARDS:-4}"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/embed.log") 2>&1
echo "[$(date -u +%FT%TZ)] multi-embed start  shards=$N  gpus=$(nvidia-smi -L 2>/dev/null | wc -l)"
nvidia-smi -L 2>/dev/null

test -d "$WORK/ingest" || { echo "FATAL: $WORK/ingest not present (corpus not delivered)"; exit 1; }

# --- venv (clean, local-matched stack; see runpod_embed.sh for the why) --------
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

# --- Qdrant (bigger request ceiling; see runpod_embed.sh) ----------------------
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

# --- embed: warm the model cache + pre-create the collection, then N sharded procs ---
cd "$WORK/ingest"
export QDRANT_URL="http://127.0.0.1:6333" COLLECTION_NAME="$COLLECTION"
export EMBED_DEVICE="cuda" EMBED_USE_FP16="true" EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-256}"
# checksum warms the HF cache (so the N shards don't race the download) + writes the GPU ref
"$PY" -m ingest embed --checksum >"$OUT/checksum_stdout.txt" 2>&1 || true
# pre-create the collection once so concurrent shards never race on creation
"$PY" -c "from ingest.config import load_config; from ingest import qdrant_store as s; \
  cfg=load_config(); c=s.make_client(cfg); s.ensure_collection(c, cfg); print('collection ready')"

# drop stale whole-source checkpoints carried in the corpus tar (shards use *.shardXofN.embed.json)
for f in "$WORK"/ingest/.state/*.embed.json; do case "$f" in *.shard*) ;; *) rm -f "$f";; esac; done 2>/dev/null || true

echo "[$(date -u +%FT%TZ)] launching $N sharded processes (one per GPU)..."
pids=""
for i in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=$i "$PY" -m ingest embed --source all --shard "$i/$N" --batch-size 256 \
    >"$OUT/shard$i.log" 2>&1 &
  pids="$pids $!"
done
fail=0
for p in $pids; do wait "$p" || fail=1; done
[ "$fail" = "0" ] || { echo "a shard FAILED — see out/shard*.log"; exit 1; }
echo "[$(date -u +%FT%TZ)] all $N shards done"

# --- persist GPU checksum + Qdrant snapshot for transfer back ------------------
"$PY" - <<'PYEOF'
import json
from ingest.config import load_config
from ingest.embedding import BGEM3Embedder
from ingest import embed_job
cfg = load_config()
sha, vec = embed_job.dense_checksum(BGEM3Embedder(cfg))
json.dump({"sha": sha, "sentence": embed_job.CHECKSUM_SENTENCE, "dense": vec},
          open("/workspace/out/checksum_gpu.json", "w"))
print("GPU checksum sha", sha)
PYEOF
SNAP=$(curl -sf -X POST "http://127.0.0.1:6333/collections/${COLLECTION}/snapshots" \
  | "$PY" -c "import sys,json; print(json.load(sys.stdin)['result']['name'])")
cp "$WORK/qdrant_snapshots/${COLLECTION}/${SNAP}" "$OUT/${COLLECTION}.snapshot"
POINTS=$(curl -sf "http://127.0.0.1:6333/collections/${COLLECTION}" \
  | "$PY" -c "import sys,json; print(json.load(sys.stdin)['result']['points_count'])")
echo "[$(date -u +%FT%TZ)] embed DONE  points=$POINTS  snapshot=$SNAP"
echo "$POINTS" > "$OUT/DONE"
