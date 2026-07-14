# RunPod Serverless hosting runbook (legal-search worker)

The whole read path (Qdrant + BGE-M3 + reranker) runs inside ONE RunPod Serverless GPU
endpoint that scales to zero — $0 compute when idle, per-second billing while a request
runs. The local MCP server becomes a thin client (`SEARCH_BACKEND=remote`); the local
Qdrant stays the write master for scraping/ingest and publishes data via
`scripts/publish_snapshot.py`. Code: `ingest/serverless/` (worker),
`ingest/ingest/remote_search.py` (client).

Provisioning is deliberately console clicks, not code: it's a one-time action and the
serverless GraphQL mutations are under-documented and Cloudflare-fronted.

## 1. One-time provisioning (console)

1. **Network volume** — Storage → New network volume:
   - Datacenter: **EU-CZ-1** (has the S3-compatible gateway and is close; fallback: US-KS-2).
   - Size: **70 GB** (~24 GB current snapshot + ~5 GB HF model cache + room for at least
     one previous immutable snapshot and headroom) ≈ **$4.90/mo** at $0.07/GB/mo.
     Restored Qdrant storage is on container disk, not this volume. Volumes can grow but not shrink.
   - Note the volume ID — it is also the S3 **bucket name**.
2. **S3 API key** — Settings → S3 API Keys → create. Record access + secret key.
3. **Build & push the image** (Docker Hub free account). This remains a release hold until
   the read-only known-good worker handoff provides all of the values below. Do not infer a
   base digest, CUDA build, torch wheel, Qdrant checksum, or dependency hash. The committed
   `serverless/requirements.txt` and `runtime-identity.unconfigured.json` are deliberately
   rejected by the production Dockerfile.

   Place the separately acquired Qdrant archive, the official version-matched checksum
   asset, complete pip `--require-hashes` lock, and validated runtime identity JSON inside
   the `ingest/` build context. The identity must include the checksum asset's SHA-256.
   Preflight them before invoking Docker; this performs only local reads and hashing:

   ```bash
   cd ingest
   python scripts/supply_chain.py validate-build-inputs \
     --base-image "$PYTHON_BASE_IMAGE" \
     --qdrant-version "$QDRANT_VER" \
     --qdrant-archive-url "$QDRANT_ARCHIVE_URL" \
     --qdrant-archive-sha256 "$QDRANT_ARCHIVE_SHA256" \
     --qdrant-checksum-source-url "$QDRANT_CHECKSUM_SOURCE_URL" \
     --qdrant-checksum-evidence "$QDRANT_CHECKSUM_EVIDENCE" \
     --qdrant-archive "$QDRANT_ARCHIVE" \
     --requirements-lock "$SERVERLESS_REQUIREMENTS_LOCK" \
     --runtime-identity "$SERVERLESS_RUNTIME_IDENTITY" \
     --known-good-worker-artifact "$KNOWN_GOOD_WORKER_ARTIFACT"

   docker build --platform linux/amd64 -f serverless/Dockerfile \
     --build-arg PYTHON_BASE_IMAGE \
     --build-arg QDRANT_VER \
     --build-arg QDRANT_ARCHIVE_URL \
     --build-arg QDRANT_ARCHIVE_SHA256 \
     --build-arg QDRANT_CHECKSUM_SOURCE_URL \
     --build-arg QDRANT_ARCHIVE \
     --build-arg QDRANT_CHECKSUM_EVIDENCE \
     --build-arg SERVERLESS_REQUIREMENTS_LOCK \
     --build-arg SERVERLESS_RUNTIME_IDENTITY \
     --build-arg KNOWN_GOOD_WORKER_ARTIFACT \
     -t "$CANDIDATE_IMAGE_TAG" .
   ```

   The runtime identity must bind the digest-pinned Python base, exact Python and
   `torch+cuNNN` versions, CUDA runtime, validated torch wheel SHA-256, complete lock-file
   SHA-256, the captured official version-specific Qdrant checksum asset and its digest,
   target
   `linux/amd64`, and the SHA-256 of the known-good worker evidence artifact. The build
   uses the locally supplied Qdrant archive and performs no archive download. It verifies
   the checksum before extraction, installs only hash-locked wheels, runs `pip check`, and
   compares installed Python/torch/CUDA identities to the evidence.

   Pushing, scanning, and deployment require separate approval. Once the exact immutable
   candidate image is already present in the local Docker daemon and Syft plus a cached
   Grype database are installed, emit owner-only evidence without pulls or database updates:

   ```bash
   python scripts/supply_chain.py emit-release-audit \
     --image "$CANDIDATE_IMAGE_DIGEST_REF" \
     --output-dir "$RELEASE_EVIDENCE_DIR"
   ```

   This writes `sbom.cdx.json`, `vulnerabilities.json`, and a checksum-bearing
   `provenance.json` into a new `0700` directory with `0600` files. It fails if the image
   digest is not already local, the scanner database is unavailable, or the destination
   exists. Review the vulnerability report as a release gate; the script does not suppress
   findings or choose a severity exception policy.
4. **Serverless endpoint** — Serverless → New Endpoint:
   - Image: `<user>/legal-search-worker:v1`.
   - GPUs: 16 GB tier (A4000 / RTX 4000 Ada / RTX 2000 Ada, $0.58/hr) with the 24 GB tier
     (L4 / A5000 / 3090, $0.69/hr) as fallback. If the probe (below) shows < ~14 GB system
     RAM on the 16 GB tier, restrict to the 24 GB tier.
   - **Max workers: 1** — HARD requirement. Qdrant holds an exclusive lock on its storage
     dir; a second worker on the same volume cannot serve and risks fighting the lock.
   - **Execution timeout: 1800 s** (the first boot after a publish sha-checks + restores
     ~24 GB — 10–20 min). Idle timeout: 60 s. FlashBoot: on.
   - Attach the network volume (mounts at `/runpod-volume`).
   - **Container disk: ≥ 80 GB** (hard requirement: ~8 GB image + ~35 GB restored Qdrant
     + recovery temp/headroom). The worker refuses to boot below 64 GB by default; keep the
     80 GB allocation until peak restore usage has been measured in the selected GPU tier.
   - Env vars (no secrets needed — Qdrant is loopback-only inside the worker, and the
     queue API is authenticated by RunPod itself):
     ```
     EMBED_DEVICE=cuda          EMBED_USE_FP16=true    EMBED_BATCH_SIZE=16
     RERANK_ENABLED=true        RERANK_DEVICE=cuda     RERANK_USE_FP16=true
     RERANK_CANDIDATES=80       RERANK_MIN_SCORE=0.3
     QUERY_LOG_ENABLED=false    HF_HOME=/runpod-volume/hf
     ```
     Do not set `COLLECTION_NAME`, `GENERATION_ID`, or model/revision identity on the
     endpoint: the worker derives and hard-binds those values from the verified publish.
     Keep retrieval-policy and chunking knobs such as `RERANK_CANDIDATES`,
     `RERANK_MIN_SCORE`, `RERANK_BACKEND`, `CITATION_ROUTE`, and `CHUNK_*` exactly aligned
     with the generation. Drift is detected against the generation fingerprint and boot
     fails closed before any model import.
5. **ingest/.env additions** (names only — values stay out of git):
   ```
   RUNPOD_ENDPOINT_ID=...      # from the endpoint page
   RUNPOD_VOLUME_ID=...        # the volume id = S3 bucket
   RUNPOD_S3_ENDPOINT=https://s3api-eu-cz-1.runpod.io
   RUNPOD_S3_REGION=eu-cz-1
   RUNPOD_S3_ACCESS_KEY=...    RUNPOD_S3_SECRET_KEY=...
   # SEARCH_BACKEND=remote     # flip AFTER --verify passes
   ```
   (`RUNPOD_API_KEY` is already there.)

## 2. Probe before committing the 24 GB upload (~$0.05)

The pattern (Qdrant co-process on serverless + network volume) is undocumented, and RunPod
doesn't publish system RAM per GPU tier. Verify for pennies, before uploading anything:

```bash
cd ingest && uv run python - <<'EOF'
from ingest.config import load_config
from ingest.remote_search import RunPodQueueClient
import json
cfg = load_config()
out = RunPodQueueClient(cfg.runpod_endpoint_id, cfg.runpod_api_key, timeout=900).call("health")
print(json.dumps(out, indent=2))
EOF
```

This cold-starts a worker against the empty volume (models download once into
`/runpod-volume/hf`, ~4.5 GB). Check in `sysinfo`: `ram_total_gb` **≥ ~14** (else restrict
the endpoint to the 24 GB tier), `volume_free_gb` sane, GPU present. `health.ok` will be
false (no collection yet) — expected.

## 3. Seeding / publishing data

Remote publishing is intentionally blocked until an immutable full-corpus generation and its
Qdrant snapshot have passed exact verification, a known-good runtime identity is recorded, and
the remote store's conditional compare-and-swap behavior has been proven. The old
`publish_snapshot.py --create` path is permanently disabled: it must never snapshot the implicit
live/legacy collection.

Prepare the local generation with `scripts/create_generation.py`, verify it with
`scripts/verify_generation.py`, and persist a `scripts/promote_generation.py plan`. Snapshot
creation and remote activation then require a separately approved generation-specific backend.
The stock publisher deliberately has no default manifest activator and therefore refuses before
S3 access. When a proven activator is supplied by deployment integration, upload and verification
still require all of `--apply`, `PUBLISH_REMOTE_APPROVED=1`, and the explicit cold-restore safety
confirmation. `--cleanup` is a plan only; deletion additionally requires `--apply` and
`ARTIFACT_PRUNE_APPROVED=1`.

- The upload prints observed Mbps after the first parts — abort early if the projection is
  unacceptable and rerun overnight (it resumes from the last completed 256 MB part).
- `manifest.json` is uploaded **last**: it is the atomic publish signal the worker's
  restore keys on. Never place a manifest for a snapshot that isn't fully uploaded.
- `--verify` uses the worker's `refresh` op and has a `PUBLISH_VERIFY_TIMEOUT` budget
  (default 1800 s), deliberately separate from the day-to-day `RUNPOD_API_TIMEOUT`. A
  process that imported search for generation A never rebinds itself to generation B: a
  refresh may stage B as its independent physical collection, but that process immediately
  abstains with `cold restart required`. Start a fresh worker, then verify its health.
- On cold boot the handler validates `manifest.json` and `ACTIVE`, the exact physical
  collection/count/schema/payload identity, and the model/vector/chunk/retrieval
  fingerprints. It hard-binds the physical collection plus immutable model revisions before
  importing the MCP search module. Ambient `COLLECTION_NAME`, model, revision, or generation
  values cannot redirect the worker.
- Before every operation the worker rechecks the current publish/ACTIVE identity and the
  live Qdrant generation. A stale result, changed publication, non-green collection, count
  drift, or payload-identity drift makes health false and every retrieval operation abstain.
  The previous generation is not silently served after publication advances.
- **Warm search-runtime replacement is deliberately refused.** For a publish today,
  temporarily disable FlashBoot, drain active work, and start a fresh worker so binding
  occurs before MCP import. Do not treat this as high-availability promotion: production
  updates still require blue/green collections, full logical verification, and an alias
  swap before switching `SEARCH_BACKEND=remote`.
- A failed cold restore makes every op return an error instead of crash-looping; fix the cause and
  retry on empty ephemeral storage.
- Republishing later means creating and verifying a new generation; immutable snapshot keys are
  never reused or overwritten.
- Steady state should move to small delta publishes (the 2.4 GB delta collection) instead
  of 24 GB fulls — not implemented yet; see plan notes.

## 4. Flip the MCP server to remote

```
ingest/.env:  SEARCH_BACKEND=remote
```
Reconnect via `/mcp`. Then smoke test: `legal_health` (endpoint health, no wake) →
`legal_collection_info` → one Georgian + one English `legal_search`; results should match
local mode exactly (same tool code runs worker-side). The local server now loads **no**
models and needs no local Qdrant — `docker compose stop` it day-to-day if you want the RAM;
start it again for ingest/publish.

Flip back anytime: `SEARCH_BACKEND=local` + `/mcp` reconnect. No silent auto-fallback —
remote errors tell the agent to retry (cold start) or suggest the env flip.

## 5. Costs & spend model

- Volume: ~$4.90/mo (70 GB × $0.07). The only recurring cost while idle.
- GPU: per-second while a worker is up: cold start 1–3 min + 60 s idle tail per burst,
  ~$0.58–0.69/hr → casual daily use ≈ $1–3/mo.
- Watch balance + endpoint state on the session monitor (`scripts/session_monitor.py`,
  http://localhost:8770) — the "Serverless GPU" tile never wakes a worker.

## 6. Teardown / emergency stop

Serverless has no orphaned-pod risk (nothing bills at idle). Three levels:
1. **Pause**: endpoint → max workers 0. $0 compute, config preserved.
2. **Delete endpoint**: console → endpoint → delete. Remote mode starts returning the
   actionable error; flip `SEARCH_BACKEND=local` to keep working.
3. **Delete the volume**: console → Storage. Stops the $4.90/mo. Do this only when the immutable
   generation snapshot, manifest, checksums, runtime identity, and rollback target are retained
   and independently verified; the mutable local Qdrant is not a reproducible backup.

Key rotation: revoke the S3 key (Settings → S3 API Keys) and/or the restricted
`RUNPOD_API_KEY`; update ingest/.env.
