---
name: georgia-legal-ops
description: Operate and evaluate the Georgia Legal Search corpus safely. Use for ingest, deployment, Qdrant health, monitoring, recovery, daily pipeline work, or retrieval-quality experiments.
---

# Georgia Legal Search Operations

Treat the repository runbooks and current working tree as authoritative. This skill guides operations; it does not authorize destructive or remote actions.

## Start here

1. Run `plugins/georgia-legal-search/scripts/health-check.sh` from the repository root.
2. Read the relevant source before proposing or running commands:
   - `ingest/docs/deployment.md` for service modes, daily ingest, monitoring, and recovery.
   - `ingest/docs/delta_embed_runbook.md` for large delta embedding.
   - `improvement.md` for experiment gates and keep-or-revert policy.
   - `HANDOFF.md` for current project state.
3. Inspect `git status` and preserve unrelated changes. Never pull or merge `origin/dev`; the project documents it as a divergent fork.

## Operational rules

- Prefer read-only checks first: `legal_health`, `ingest_status`, Qdrant collection state, container status, logs, and checkpoint inspection.
- Do not create `.env`, install dependencies, download models, start containers, mutate Qdrant, scrape, ingest, re-embed, deploy, or terminate remote resources without an explicit user request.
- Never expose Qdrant publicly. Require an API key for non-local Qdrant and HTTPS when transmitting it.
- Run one watcher per collection. Respect lock guards and graceful shutdown paths.
- Treat yellow index state during writes as expected but unsuitable for hybrid queries; verify green state before declaring serving healthy.
- Avoid `--recreate`, collection deletion, checkpoint removal, or state-file edits unless the user explicitly authorizes the destructive operation and recovery impact is understood.

## Retrieval evaluation

- Keep golden set v1 frozen. Add new judgments to the versioned successor rather than rewriting v1.
- Compare configurations at the same corpus state and record `points_count`.
- Use the repository harness, bootstrap confidence intervals, paired tests, and BM25 floor described in `improvement.md`.
- Keep serving `retrieval_fingerprint` distinct from evaluation `config_hash`.
- Report exact commands, configuration, corpus state, metrics, and whether the documented promotion gate passed.

## Completion evidence

For operational work, report the checks run, observable state before and after, changed resources, and remaining recovery steps. A command exiting successfully is not sufficient if collection health, coverage, or evaluation artifacts contradict it.
