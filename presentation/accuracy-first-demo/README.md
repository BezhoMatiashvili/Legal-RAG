# Accuracy-first Georgian legal demonstration

This is an accuracy-first pre-production Georgian legal RAG. It answers only when evidence, identity, version, quotation, and completeness checks pass; otherwise it clarifies or abstains.

This frozen bundle separates three tracks: model-free read-only exact lookup against a
local legacy collection; offline span/hash checks against frozen snapshots; and synthetic
schema-v2 evidence-contract fixtures. None is described as a production legal-answer run.

## Rehearse offline

From the repository root:

```bash
ingest/.venv/bin/python ingest/scripts/build_presentation_demo.py rehearse \
  --bundle presentation/accuracy-first-demo
```

The rehearsal verifies all persisted hashes and prints `demo-script.md`. It does not access
Qdrant, load a model, or use the network.

## Verify only

```bash
ingest/.venv/bin/python ingest/scripts/build_presentation_demo.py verify \
  --bundle presentation/accuracy-first-demo
```

## Track labels

- `legacy_read_only_exact_selector_scroll`: measured exact identifier retrieval only.
- `offline_frozen_snapshot_audit_not_legacy_retrieval_evidence`: local quote/offset/body hash checks.
- `synthetic_contract_fixture`: synthetic evidence-schema rejection/acceptance checks, excluded from retrieval results.
- Production target: a sealed generation with pinned runtime identity, freshness, and release evidence.

Legal correctness review: not yet independently adjudicated.
