#!/usr/bin/env python3
"""RunPod Serverless queue handler for the Georgian legal RAG ("legal-search worker").

One worker = the whole read path: a local Qdrant (restored onto ephemeral container disk), BGE-M3
and the cross-encoder reranker on GPU, and an op dispatcher that calls the MCP tool
coroutines from ``ingest.mcp_server`` directly. ``FastMCP.tool()`` returns the original
function, so ``legal_search`` etc. are plain awaitables here and every op returns the
same formatted string the local MCP server would produce — the local server in
``SEARCH_BACKEND=remote`` mode just passes it through.

Request shape (queue API ``input``):   {"op": "search", "params": {...}}
Response shape (job ``output``):       {"result": "<tool output string>"}   or {"error": ...}

Ops: ask | get_context | search | get_document | lookup | browse | versions |
collection_info | health | refresh.
``ingest_status`` is intentionally absent — watcher state lives on the local box only.

Boot is FAIL-SOFT: an exception during qdrant boot/restore must not exit the process —
RunPod would restart the container in a billed crash-loop. Instead the error is stored and
every op reports it; the ``refresh`` op retries the boot.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import traceback

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("serverless.handler")

# The worker IS the backend: never let a stray endpoint env var route it to itself, and
# never write query logs off-box (hard assign — privacy guardrail, not a default).
os.environ["SEARCH_BACKEND"] = "local"
os.environ["QUERY_LOG_ENABLED"] = "false"
os.environ.setdefault("HF_HOME", "/runpod-volume/hf")

import qdrant_boot  # noqa: E402  (sibling module; script dir is on sys.path)

_BOOT_ERROR: str | None = None
_RESTORE: dict = {"status": "not_checked"}
_RUNTIME_MANIFEST: dict | None = None
_RUNTIME_CFG = None


def _configure_runtime_environment(manifest: dict):
    """Bind immutable worker config and prove its fingerprints before MCP import."""
    identity = manifest["point_identity"]
    vector = manifest["vector_space"]
    forced = {
        "SEARCH_BACKEND": "local",
        "QUERY_LOG_ENABLED": "false",
        "QDRANT_URL": qdrant_boot.QDRANT_URL,
        "COLLECTION_NAME": manifest["collection"],
        "GENERATION_ID": manifest["generation_id"],
        "EMBED_MODEL": identity["embedding_model"],
        "EMBED_REVISION": identity["embedding_revision"],
        "TOKENIZER_MODEL": identity["tokenizer_model"],
        "TOKENIZER_REVISION": identity["tokenizer_revision"],
        "RERANK_MODEL": identity["reranker_model"],
        "RERANK_REVISION": identity["reranker_revision"],
        # The worker owns the pinned local reranker; an ambient URL could silently route
        # production scoring to an unverified external model with the same display name.
        "RERANK_REMOTE_URL": "",
        "DENSE_DIM": str(vector["dense_dimension"]),
        "EMBED_HEADER_V2": "true" if identity["document_header"] else "false",
        # Load the environment config in development mode, then elevate the frozen
        # instance only after the handler has verified every generation identity below.
        "PRODUCTION_MODE": "false",
    }
    os.environ.update(forced)
    # A machine/image path cannot substitute for the verified serverless boot protocol.
    # The in-process MCP readiness probe installed below is the only permitted exception
    # to the normal full GENERATION_DIR artifact requirement.
    os.environ.pop("GENERATION_DIR", None)
    os.environ.pop("VERIFIED_WORKER_BINDING", None)

    from dataclasses import replace

    from ingest.config import load_config, validate_production_config
    from ingest.qdrant_store import generation_point_identity

    cfg = replace(
        load_config(), production_mode=True, verified_worker_binding=True
    )
    validate_production_config(cfg)
    actual_identity = generation_point_identity(cfg)
    if actual_identity is None:
        raise RuntimeError("bound worker configuration has no generation identity")
    actual_payload = actual_identity.as_payload()
    identity_mismatches = sorted(
        key
        for key, expected in identity.items()
        if actual_payload.get(key) != expected
    )
    worker_vector = {
        "dense_name": "dense",
        "dense_dimension": cfg.dense_dim,
        "distance": "cosine",
        "sparse_name": "sparse",
    }
    vector_mismatches = sorted(
        key for key, expected in vector.items() if worker_vector.get(key) != expected
    )
    if identity_mismatches or vector_mismatches:
        fields = [
            *(f"point_identity.{key}" for key in identity_mismatches),
            *(f"vector_space.{key}" for key in vector_mismatches),
        ]
        raise RuntimeError(
            "worker search configuration does not match restored generation: "
            + ", ".join(fields)
        )
    # Any accidental later load_config() call now fails closed without GENERATION_DIR;
    # mcp_server receives the already-validated frozen cfg through its installer.
    os.environ["PRODUCTION_MODE"] = "true"
    return cfg


def _accept_restore(candidate: dict, *, allow_initial_bind: bool) -> None:
    """Accept only the cold-bound identity; warm generation changes need a restart."""
    global _RESTORE, _RUNTIME_CFG, _RUNTIME_MANIFEST
    _RESTORE = candidate
    manifest = qdrant_boot.verified_runtime_manifest(candidate)
    if allow_initial_bind:
        _RUNTIME_CFG = _configure_runtime_environment(manifest)
        _RUNTIME_MANIFEST = manifest
        return
    if _RUNTIME_MANIFEST is None:
        raise RuntimeError(
            "generation became ready after MCP import; cold restart required before serving"
        )
    if (
        qdrant_boot.runtime_manifest_identity(manifest)
        != qdrant_boot.runtime_manifest_identity(_RUNTIME_MANIFEST)
    ):
        raise RuntimeError(
            "published generation changed after MCP import; cold restart required before serving"
        )


def _boot(*, allow_initial_bind: bool = False) -> None:
    """Cold-start boot: qdrant up + apply any pending publish. Sets globals, never raises."""
    global _BOOT_ERROR, _RESTORE
    try:
        qdrant_boot.ensure_running()
        candidate = qdrant_boot.maybe_restore()
        if candidate.get("status") == "no_manifest":
            # A newly provisioned worker may be probed before any approved generation is
            # published. Keep liveness/health available, but bind nothing and serve no ops.
            _RESTORE = candidate
            _BOOT_ERROR = None
            logger.info("boot restore check: %s", _RESTORE)
            return
        _accept_restore(candidate, allow_initial_bind=allow_initial_bind)
        logger.info("boot restore check: %s", _RESTORE)
        _BOOT_ERROR = None
    except Exception as e:  # noqa: BLE001 - see module docstring: no billed crash-loops
        _BOOT_ERROR = f"{type(e).__name__}: {e}"
        logger.error("worker boot failed: %s\n%s", _BOOT_ERROR, traceback.format_exc())


_boot(allow_initial_bind=True)

from ingest import mcp_server as srv  # noqa: E402  (import is side-effect-free)


def _worker_readiness() -> dict:
    return qdrant_boot.runtime_readiness(_RESTORE, _RUNTIME_MANIFEST)


def _retrieval_license_attestations():
    """Load deployment-reviewed attestations before either retrieval model can load."""

    from ingest.model_policy import LicenseAttestation

    paths = {
        "retriever": os.getenv("RETRIEVER_LICENSE_ATTESTATION_PATH", "").strip(),
        "reranker": os.getenv("RERANKER_LICENSE_ATTESTATION_PATH", "").strip(),
    }
    missing = sorted(name for name, path in paths.items() if not path)
    if missing:
        raise RuntimeError(
            "production worker is missing retrieval license attestation paths: "
            + ", ".join(missing)
        )
    return (
        LicenseAttestation.from_path(paths["retriever"]),
        LicenseAttestation.from_path(paths["reranker"]),
    )


if _RUNTIME_CFG is not None:
    try:
        retriever_attestation, reranker_attestation = _retrieval_license_attestations()
        srv._install_verified_worker_runtime(
            _RUNTIME_CFG,
            _worker_readiness,
            retriever_attestation=retriever_attestation,
            reranker_attestation=reranker_attestation,
        )
    except Exception as e:  # noqa: BLE001 - keep fail-soft boot, but never serve unbound
        _BOOT_ERROR = f"{type(e).__name__}: {e}"
        logger.error("worker runtime binding install failed: %s", _BOOT_ERROR)

OPS: dict[str, tuple] = {
    "ask": (srv.legal_ask, srv.LegalAskInput),
    "get_context": (srv.legal_get_context, srv.LegalGetContextInput),
    "search": (srv.legal_search, srv.SearchInput),
    "get_document": (srv.legal_get_document, srv.GetDocumentInput),
    "lookup": (srv.legal_lookup, srv.LookupInput),
    "browse": (srv.legal_browse, srv.BrowseInput),
    "versions": (srv.legal_get_document_versions, srv.GetVersionsInput),
    "collection_info": (srv.legal_collection_info, None),
    "health": (srv.legal_health, None),
}


def _sysinfo() -> dict:
    """RAM/CPU/disk/GPU facts for the probe step — the 16GB-VRAM tier's system RAM is
    undocumented, and Qdrant + both models need ~14GB; this makes the fit measurable."""
    info: dict = {"cpus": os.cpu_count()}
    try:
        mem = {}
        for ln in open("/proc/meminfo"):
            k, _, v = ln.partition(":")
            mem[k.strip()] = int(v.strip().split()[0])
        info["ram_total_gb"] = round(mem.get("MemTotal", 0) / 1024 / 1024, 1)
        info["ram_available_gb"] = round(mem.get("MemAvailable", 0) / 1024 / 1024, 1)
    except OSError:
        pass
    try:
        du = shutil.disk_usage(qdrant_boot.VOLUME_ROOT)
        info["volume_total_gb"] = round(du.total / 1e9, 1)
        info["volume_free_gb"] = round(du.free / 1e9, 1)
    except OSError:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["vram_total_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    except Exception:  # noqa: BLE001 - sysinfo must never fail an op
        pass
    return info


async def _warmup() -> None:
    """Load both models eagerly. Init time is billed either way, and FlashBoot then
    snapshots a worker with everything resident, so warm revivals answer instantly and
    the first search never pays the model load inside its own request."""
    readiness = _worker_readiness()
    if not readiness.get("ok"):
        raise RuntimeError(
            "refusing model warmup without exact generation readiness: "
            f"{readiness.get('code', 'unknown')}"
        )
    await srv._get_embedder()
    await srv._get_reranker()


async def handler(job: dict) -> dict:
    global _BOOT_ERROR, _RESTORE
    inp = job.get("input") or {}
    op = inp.get("op") or "health"
    params = inp.get("params") or {}
    try:
        if op == "refresh":
            # Retry a failed boot, else re-check the published manifest — both without
            # waiting for the next cold start (FlashBoot revivals skip module import).
            if _BOOT_ERROR is not None:
                await asyncio.to_thread(_boot)
                if _BOOT_ERROR is not None:
                    return {
                        "error": f"worker boot failed: {_BOOT_ERROR}",
                        "restore": _RESTORE,
                    }
            else:
                candidate = await asyncio.to_thread(
                    qdrant_boot.maybe_restore, bool(params.get("force"))
                )
                try:
                    _accept_restore(candidate, allow_initial_bind=False)
                except Exception as e:  # noqa: BLE001 - warm identity changes fail closed
                    _BOOT_ERROR = f"{type(e).__name__}: {e}"
                    return {
                        "error": f"worker boot failed: {_BOOT_ERROR}",
                        "restore": _RESTORE,
                    }
            if _RESTORE.get("status") == "restored":
                srv._result_cache.clear()
            return {"result": json.dumps(_RESTORE, ensure_ascii=False), "restore": _RESTORE}

        if _BOOT_ERROR is not None:
            return {"error": f"worker boot failed: {_BOOT_ERROR}. "
                             "Send op 'refresh' to retry the boot."}

        # A publish that landed after module import is caught by a cheap two-file check.
        # maybe_restore refuses destructive warm replacement, so this records a visible
        # stale-data warning until an empty cold worker (or blue/green promotion) is used.
        # ``restore_pending`` is only two small local JSON reads. Running it directly
        # avoids creating a per-request default executor (and makes pre-publication health
        # probes terminate cleanly); the potentially multi-minute restore below remains
        # offloaded.
        if qdrant_boot.restore_pending():
            logger.info("new publish detected on a warm worker; checking safe restore policy")
            candidate = await asyncio.to_thread(qdrant_boot.maybe_restore)
            try:
                _accept_restore(candidate, allow_initial_bind=False)
            except Exception as e:  # noqa: BLE001 - never mix boot and imported identities
                _BOOT_ERROR = f"{type(e).__name__}: {e}"
                return {
                    "error": f"worker boot failed: {_BOOT_ERROR}",
                    "restore": _RESTORE,
                }
            if _RESTORE.get("status") == "restored":
                srv._result_cache.clear()

        readiness = _worker_readiness()
        generation_ready = bool(readiness.get("ok"))
        if op == "health" and not generation_ready:
            health = {
                "ok": False,
                "code": readiness.get("code", "generation_not_verified"),
                "restore_status": _RESTORE.get("status", "unknown"),
                "restore_code": _RESTORE.get("code", "unverified"),
                "readiness": readiness,
            }
            return {
                "result": json.dumps(health, ensure_ascii=False),
                "sysinfo": _sysinfo(),
                "restore": _RESTORE,
            }
        if op != "health" and not generation_ready:
            return {
                "error": (
                    "worker abstained: no exact, green, immutable generation is verified; "
                    f"restore_status={_RESTORE.get('status', 'unknown')!r} "
                    f"restore_code={_RESTORE.get('code', 'unverified')!r}"
                ),
                "restore": _RESTORE,
            }

        entry = OPS.get(op)
        if entry is None:
            return {"error": f"unknown op {op!r}; expected one of "
                             f"{sorted([*OPS, 'refresh'])}"}
        fn, model = entry
        result = await fn(model(**params)) if model is not None else await fn()
        out = {"result": result}
        if op == "health":
            out["sysinfo"] = _sysinfo()
            out["restore"] = _RESTORE
        return out
    except Exception as e:  # noqa: BLE001 - the job output is the only error channel
        tb = traceback.format_exc().splitlines()[-12:]
        logger.error("op %s failed: %s\n%s", op, e, "\n".join(tb))
        return {"error": f"{type(e).__name__}: {e}", "traceback": tb}


def main() -> None:
    if os.getenv("EAGER_LOAD", "1").strip().lower() not in {"0", "false", "no", "off"}:
        try:
            asyncio.run(_warmup())
            logger.info("models warm; sysinfo=%s", _sysinfo())
        except Exception as e:  # noqa: BLE001 - lazy load retries per search; don't crash-loop
            logger.error("eager model load failed (will retry lazily): %s", e)

    import runpod

    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
