"""Test-suite environment isolation.

The test suite pins the load-bearing invariants (point-id determinism, retrieval
fingerprint, remote-op wire shape, …) and must be reproducible regardless of how this
particular box is configured to *serve*. But `ingest/ingest/config.py` calls
``load_dotenv()`` at import time (``override=False`` — see the ``env-at-spawn`` contract in
memory-bank/contracts.md), so a developer's live deployment ``ingest/.env`` leaks into the
test process. On a box configured for remote/int8 serving that flips several tools onto
their committed *remote* branch and folds the citation-route knob into the fingerprint,
producing spurious failures (``legal_health`` KeyError ``'points'``,
``get_document_versions`` JSON errors, the I4 fingerprint mismatch) that do NOT reflect any
code or test defect — a clean CI checkout (no ``.env``) is green.

This conftest makes a bare ``pytest`` match CI by pinning exactly the three serving knobs
that diverge from the committed defaults, *before* any test module imports
``ingest.config``. ``load_dotenv(override=False)`` then yields to these already-set values.

``RERANK_ENABLED`` is also pinned to ``false`` — matching CI (``.github/workflows/ci.yml``,
which sets it for exactly this reason) and the ``ram-discipline`` contract. ``rerank_enabled``
defaults **True** and the live ``.env`` may set it, so without this pin a bare ``pytest``
would let a reranker-exercising test instantiate the ~2.3 GB cross-encoder against the
resident ~21 GB Qdrant on this 30 GB box → swap-thrash. This is a test-process env var only;
RAG *serving* is unaffected (the serving process reads its own ``.env`` at spawn).

Scope is deliberately minimal: these are the only vars that diverge from CI; retrieval knobs
(``COLLECTION_NAME``/``EMBED_MODEL``/``DENSE_DIM``/``CHUNK_*``/``RERANK_MODEL`` …) are left
untouched so this changes nothing about what the tests actually exercise. Individual tests
may still opt into other modes via ``monkeypatch.setenv`` (tests re-load config per call).
"""

import os

# Committed defaults (config.py): SEARCH_BACKEND→"local", RERANK_BACKEND→"torch",
# CITATION_ROUTE→"" maps to None (off) via _route_opt. RERANK_ENABLED→"false" mirrors CI's
# ram-discipline pin. Forced (not setdefault) so the suite is hermetic and equals CI even
# when the shell/.env selects a different serving mode.
os.environ["SEARCH_BACKEND"] = "local"
os.environ["RERANK_BACKEND"] = "torch"
os.environ["CITATION_ROUTE"] = ""
os.environ["RERANK_ENABLED"] = "false"
