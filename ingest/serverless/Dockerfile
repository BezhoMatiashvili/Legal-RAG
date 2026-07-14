# RunPod Serverless "legal-search worker": Qdrant (static binary, restored storage on the
# ephemeral container disk) + BGE-M3 + bge-reranker-v2-m3 + an op-dispatch queue handler that reuses the
# MCP tool coroutines from ingest/ingest/mcp_server.py unchanged.
#
# Build from ingest/ only after scripts/supply_chain.py validates the observed release
# inputs. There are deliberately no defaults for the base digest, Qdrant checksum, or
# validated runtime identity; docs/runpod-serverless.md contains the fail-closed command.
#
# The image bakes code + binaries only. Published snapshots and the HF model cache live on
# the network volume mounted at /runpod-volume; restored Qdrant data lives on container disk. Secrets never enter
# the image (ingest/.dockerignore excludes .env; the queue API is authed by RunPod itself).
ARG PYTHON_BASE_IMAGE
FROM --platform=linux/amd64 ${PYTHON_BASE_IMAGE}

ARG PYTHON_BASE_IMAGE
ARG QDRANT_VER
ARG QDRANT_ARCHIVE_URL
ARG QDRANT_ARCHIVE_SHA256
ARG QDRANT_CHECKSUM_SOURCE_URL
# These defaults are intentionally invalid production inputs. They keep the failure
# actionable while requiring callers to supply observed artifacts from the build context.
ARG QDRANT_ARCHIVE=serverless/runtime-identity.unconfigured.json
ARG QDRANT_CHECKSUM_EVIDENCE=serverless/runtime-identity.unconfigured.json
ARG SERVERLESS_REQUIREMENTS_LOCK=serverless/requirements.txt
ARG SERVERLESS_RUNTIME_IDENTITY=serverless/runtime-identity.unconfigured.json
ARG KNOWN_GOOD_WORKER_ARTIFACT=serverless/runtime-identity.unconfigured.json

COPY scripts/supply_chain.py /opt/release/supply_chain.py
COPY ${QDRANT_ARCHIVE} /opt/release/qdrant.tar.gz
COPY ${QDRANT_CHECKSUM_EVIDENCE} /opt/release/qdrant-checksums.txt
COPY ${SERVERLESS_REQUIREMENTS_LOCK} /opt/release/requirements.lock
COPY ${SERVERLESS_RUNTIME_IDENTITY} /opt/release/runtime-identity.json
COPY ${KNOWN_GOOD_WORKER_ARTIFACT} /opt/release/known-good-worker-artifact.json

# Validate every externally observed identity before extracting or installing anything.
# The archive is supplied from the build context, so this build performs no download.
RUN python /opt/release/supply_chain.py validate-build-inputs \
      --base-image "${PYTHON_BASE_IMAGE}" \
      --qdrant-version "${QDRANT_VER}" \
      --qdrant-archive-url "${QDRANT_ARCHIVE_URL}" \
      --qdrant-archive-sha256 "${QDRANT_ARCHIVE_SHA256}" \
      --qdrant-checksum-source-url "${QDRANT_CHECKSUM_SOURCE_URL}" \
      --qdrant-checksum-evidence /opt/release/qdrant-checksums.txt \
      --qdrant-archive /opt/release/qdrant.tar.gz \
      --requirements-lock /opt/release/requirements.lock \
      --runtime-identity /opt/release/runtime-identity.json \
      --known-good-worker-artifact /opt/release/known-good-worker-artifact.json \
    && mkdir -p /opt/qdrant /app/serverless \
    && tar xzf /opt/release/qdrant.tar.gz -C /opt/qdrant \
         --no-same-owner --no-same-permissions \
    && chmod 0555 /opt/qdrant/qdrant \
    && /opt/qdrant/qdrant --version \
    && install -m 0444 /opt/release/requirements.lock /app/serverless/requirements.lock \
    && install -m 0444 /opt/release/runtime-identity.json /app/serverless/runtime-identity.json

# --require-hashes makes every direct and transitive Python artifact explicit. The base
# digest fixes pip itself; no upgrade or resolver-generated dependency may float here.
RUN pip install --no-cache-dir --require-hashes --only-binary=:all: \
      -r /app/serverless/requirements.lock \
    && pip check \
    && python /opt/release/supply_chain.py verify-installed-runtime \
      --requirements-lock /app/serverless/requirements.lock \
      --runtime-identity /app/serverless/runtime-identity.json

COPY ingest/ /app/ingest/
COPY serverless/handler.py serverless/qdrant_boot.py /app/serverless/

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    RUNTIME_IDENTITY_PATH=/app/serverless/runtime-identity.json

LABEL org.opencontainers.image.base.name="${PYTHON_BASE_IMAGE}" \
      org.georgia-legal-search.qdrant.version="${QDRANT_VER}" \
      org.georgia-legal-search.qdrant.archive.sha256="${QDRANT_ARCHIVE_SHA256}"

WORKDIR /app
CMD ["python", "-u", "/app/serverless/handler.py"]
