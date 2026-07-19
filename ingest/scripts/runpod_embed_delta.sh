#!/usr/bin/env bash
# One-pod, one-source BGE-M3 delta embed. The host supplies a strict input manifest and a
# run-scoped collection name. This script proves GPU checksum parity before creating any
# corpus vectors, then emits a self-contained run manifest and immutable snapshot.
set -euo pipefail

if [ "${RUNPOD_EPHEMERAL_QDRANT:-0}" != "1" ]; then
  echo "refusing delta embed outside an explicitly attested ephemeral RunPod Qdrant" >&2
  exit 78
fi
if [ "${QDRANT_WRITE_APPROVED:-0}" != "1" ] || \
   [ "${QDRANT_RECREATE_APPROVED:-0}" != "1" ]; then
  echo "refusing delta embed without explicit write and recreate approvals" >&2
  exit 78
fi

WORK="${WORK:-/workspace}"
OUT="$WORK/out"
COLLECTION="${COLLECTION_NAME:?COLLECTION_NAME is required}"
SOURCE="${SOURCE_NAME:?SOURCE_NAME is required}"
RUN="${RUN_ID:?RUN_ID is required}"
QDRANT_VER="${QDRANT_VER:-v1.18.2}"
EXPECTED="$WORK/delta_input_manifest.json"

case "$COLLECTION" in
  "georgian_legal_delta_${SOURCE}_${RUN}") ;;
  *) echo "unsafe/non-run-scoped collection: $COLLECTION" >&2; exit 1 ;;
esac

mkdir -p "$OUT"
rm -f "$OUT"/{DONE,checksum_gpu.json,input_manifest_gpu.json,embed_report.json,run_manifest.json}
exec > >(tee -a "$OUT/embed.log") 2>&1

EXPECTED_GPU="${EXPECTED_GPU:-NVIDIA GeForce RTX 4090}"
GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
GPU_COUNT="$(printf '%s\n' "$GPU_NAMES" | sed '/^[[:space:]]*$/d' | wc -l)"
if [ "$GPU_COUNT" -ne 1 ] || [ "$GPU_NAMES" != "$EXPECTED_GPU" ]; then
  echo "GPU attestation failed: expected=$EXPECTED_GPU count=$GPU_COUNT names=$GPU_NAMES" >&2
  exit 1
fi
echo "[$(date -u +%FT%TZ)] delta start source=$SOURCE run=$RUN gpu=$GPU_NAMES"

# 1. Decrypt the exact host-validated payload. Keep the venv across an in-pod retry, but
# replace code/input so stale files can never enter this run.
cd "$WORK"
test -n "${PASSPHRASE:-}" || { echo "PASSPHRASE not set" >&2; exit 1; }
rm -rf "$WORK/ingest" "$WORK/delta_items" "$EXPECTED"
openssl enc -d -aes-256-cbc -pbkdf2 -pass env:PASSPHRASE -in payload.tar.gz.enc \
  | tar xz --no-same-owner --no-same-permissions
test -s "$EXPECTED" || { echo "host input manifest missing" >&2; exit 1; }
ls "$WORK"/delta_items/*.jsonl >/dev/null 2>&1 \
  || { echo "no delta_items/*.jsonl in payload" >&2; exit 1; }

# 2. Clean pinned environment, CUDA only. Reranking is explicitly disabled for RAM safety.
PY="$WORK/venv/bin/python"
if [ ! -x "$PY" ]; then python -m venv "$WORK/venv"; fi
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q torch --index-url https://download.pytorch.org/whl/cu124
"$PY" -m pip install -q "FlagEmbedding==1.4.0" "transformers==5.12.1" \
  "tokenizers==0.22.2" "accelerate==1.14.0" "datasets==5.0.0" \
  "sentencepiece==0.2.1" "safetensors==0.8.0" "qdrant-client>=1.12" \
  "python-dotenv>=1.0" "tqdm>=4.66" "rich>=13"
"$PY" -c "import os, torch; assert torch.cuda.is_available(), 'no CUDA'; \
assert torch.cuda.device_count() == 1; \
expected = os.environ.get('EXPECTED_GPU', 'NVIDIA GeForce RTX 4090'); \
actual = torch.cuda.get_device_name(0); \
assert actual == expected, f'CUDA device {actual!r} != expected {expected!r}'; \
print('CUDA attested:', torch.__version__, torch.version.cuda, actual)"

cd "$WORK/ingest"
unset GENERATION_ID GENERATION_DIR
export PRODUCTION_MODE="false"
export QDRANT_URL="http://127.0.0.1:6333" COLLECTION_NAME="$COLLECTION"
export EMBED_DEVICE="cuda" EMBED_USE_FP16="true" EMBED_BATCH_SIZE="256"
export RERANK_ENABLED="false"

# 3. Reproduce the host's exact IDs/chunk count on the pod before loading the encoder.
"$PY" scripts/embed_delta.py --source "$SOURCE" --items "$WORK"/delta_items/*.jsonl \
  --collection "$COLLECTION" --dry-run --strict --expect-manifest "$EXPECTED" \
  --manifest-out "$OUT/input_manifest_gpu.json"

# 4. G2 gate BEFORE corpus embedding. The checksum is a separate durable output file; a
# mismatch exits here, before Qdrant or any delta point is created.
"$PY" - <<'PYEOF'
import json
import os
from pathlib import Path

from ingest.config import load_config
from ingest.embedding import BGEM3Embedder
from ingest import embed_job

out = Path("/workspace/out/checksum_gpu.json")
cpu_path = Path("/workspace/ingest/snapshots/v1/checksum_cpu.json")
cpu = json.loads(cpu_path.read_text(encoding="utf-8"))
cfg = load_config()
sha, vec = embed_job.dense_checksum(BGEM3Embedder(cfg))
if len(cpu.get("dense") or []) != 1024 or len(vec) != 1024:
    raise SystemExit("checksum dimension mismatch")
if cpu.get("sentence") != embed_job.CHECKSUM_SENTENCE:
    raise SystemExit("CPU checksum sentence mismatch")
cosine = embed_job.checksum_cosine(cpu["dense"], vec)
payload = {
    "sha": sha,
    "sentence": embed_job.CHECKSUM_SENTENCE,
    "dense": vec,
    "cosine": cosine,
    "gate": 0.999,
    "gpu": os.environ.get("EXPECTED_GPU", "NVIDIA GeForce RTX 4090"),
}
tmp = out.with_suffix(".tmp")
tmp.write_text(json.dumps(payload), encoding="utf-8")
os.replace(tmp, out)
print(f"GPU checksum cosine={cosine:.6f}")
if cosine < 0.999:
    raise SystemExit(f"GPU checksum gate failed: {cosine:.6f} < 0.999")
PYEOF

# 5. Start fresh pod-local Qdrant only after G2 passes.
cd "$WORK"
if ! curl -sf http://127.0.0.1:6333/ >/dev/null 2>&1; then
  curl -sSL -o qdrant.tar.gz \
    "https://github.com/qdrant/qdrant/releases/download/${QDRANT_VER}/qdrant-x86_64-unknown-linux-musl.tar.gz"
  tar xzf qdrant.tar.gz --no-same-owner --no-same-permissions
  rm -rf "$WORK/qdrant_storage" "$WORK/qdrant_snapshots/$COLLECTION"
  ( QDRANT__STORAGE__STORAGE_PATH="$WORK/qdrant_storage" \
    QDRANT__STORAGE__SNAPSHOTS_PATH="$WORK/qdrant_snapshots" \
    QDRANT__SERVICE__MAX_REQUEST_SIZE_MB=1024 \
    ./qdrant >"$OUT/qdrant.log" 2>&1 & )
  for _ in $(seq 1 60); do
    curl -sf http://127.0.0.1:6333/ >/dev/null 2>&1 && break
    sleep 1
  done
fi
curl -sf http://127.0.0.1:6333/ >/dev/null \
  || { echo "Qdrant did not become ready" >&2; exit 1; }

# 6. Embed strictly into the run-scoped collection. Recreate makes retries exact rather
# than accidentally retaining points from an earlier attempt.
cd "$WORK/ingest"
"$PY" scripts/embed_delta.py --source "$SOURCE" --items "$WORK"/delta_items/*.jsonl \
  --collection "$COLLECTION" --batch-size 256 --strict --recreate --apply \
  --expect-manifest "$EXPECTED" --manifest-out "$OUT/embed_report.json"

# 7. Snapshot, hash, and write the final run manifest atomically before DONE.
SNAP_NAME="$(curl -sf -X POST \
  "http://127.0.0.1:6333/collections/${COLLECTION}/snapshots" \
  | "$PY" -c "import sys,json; print(json.load(sys.stdin)['result']['name'])")"
cp "$WORK/qdrant_snapshots/${COLLECTION}/${SNAP_NAME}" "$OUT/${COLLECTION}.snapshot"
POINTS="$(curl -sf "http://127.0.0.1:6333/collections/${COLLECTION}" \
  | "$PY" -c "import sys,json; print(json.load(sys.stdin)['result']['points_count'])")"

SOURCE_NAME="$SOURCE" RUN_ID="$RUN" POINTS="$POINTS" SNAPSHOT="$OUT/${COLLECTION}.snapshot" \
  "$PY" - <<'PYEOF'
import hashlib
import json
import os
from pathlib import Path

out = Path("/workspace/out")
expected = json.loads(Path("/workspace/delta_input_manifest.json").read_text(encoding="utf-8"))
embedded = json.loads((out / "embed_report.json").read_text(encoding="utf-8"))
checksum = json.loads((out / "checksum_gpu.json").read_text(encoding="utf-8"))
snapshot = Path(os.environ["SNAPSHOT"])
digest = hashlib.sha256()
with snapshot.open("rb") as fh:
    for block in iter(lambda: fh.read(16 * 1024 * 1024), b""):
        digest.update(block)
points = int(os.environ["POINTS"])
if points != embedded["chunks"] or points != embedded["points_count"]:
    raise SystemExit(
        f"point mismatch: qdrant={points} chunks={embedded['chunks']} "
        f"report={embedded['points_count']}"
    )
document_ids = embedded.get("document_ids")
if (
    not isinstance(document_ids, list)
    or not document_ids
    or not all(isinstance(value, str) and value for value in document_ids)
):
    raise SystemExit("embed report has no document_ids")
canonical_ids = sorted(set(document_ids))
if document_ids != canonical_ids:
    raise SystemExit("embed report document_ids are not sorted and unique")
ids_sha256 = hashlib.sha256("\n".join(document_ids).encode("utf-8")).hexdigest()
if ids_sha256 != embedded.get("document_ids_sha256"):
    raise SystemExit("embed report document_ids_sha256 is not canonical")
if (
    document_ids != expected.get("document_ids")
    or ids_sha256 != expected.get("document_ids_sha256")
):
    raise SystemExit("embedded document identities differ from the staged input manifest")
manifest = {
    "schema_version": 1,
    "source": os.environ["SOURCE_NAME"],
    "run_id": os.environ["RUN_ID"],
    "collection": embedded["collection"],
    "input_sha256": expected["input_sha256"],
    "document_ids": document_ids,
    "document_ids_sha256": ids_sha256,
    "expected_documents": expected["documents"],
    "documents": embedded["documents"],
    "expected_chunks": expected["chunks"],
    "chunks": embedded["chunks"],
    "skipped": embedded["skipped"],
    "points_count": points,
    "checksum_cosine": checksum["cosine"],
    "gpu": checksum["gpu"],
    "snapshot": snapshot.name,
    "snapshot_size_bytes": snapshot.stat().st_size,
    "snapshot_sha256": digest.hexdigest(),
}
target = out / "run_manifest.json"
tmp = target.with_suffix(".tmp")
tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
os.replace(tmp, target)
print(json.dumps(manifest, ensure_ascii=False))
PYEOF

echo "[$(date -u +%FT%TZ)] delta DONE source=$SOURCE docs/chunks validated points=$POINTS"
echo "ok" > "$OUT/DONE"
