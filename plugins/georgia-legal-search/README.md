# Georgia Legal Search Codex plugin

This repo-local plugin exposes the existing `legal_rag` MCP server and adds maintainer workflows for grounded legal research, corpus operations, and retrieval evaluation.

## Prerequisites

- A local clone of this repository with `uv` available.
- `uv sync` completed in `ingest/`.
- `ingest/.env` configured for the intended Qdrant collection.
- Qdrant running when search or collection tools are used. The plugin does not start it automatically.

From the repository root, run the read-only diagnostic:

```bash
plugins/georgia-legal-search/scripts/health-check.sh
```

Warnings report optional or currently stopped services; failures identify prerequisites that prevent the MCP server from loading.

## Install in Codex

This is a non-default, repository-local marketplace. From the repository root, register the marketplace and install the plugin:

```bash
codex plugin marketplace add "$PWD"
codex plugin add georgia-legal-search@georgia-legal-search
```

Start a new Codex thread after installation so the MCP tools and skills are loaded.

## Local updates

After changing the plugin, use the plugin-creator cachebuster helper rather than editing marketplace configuration by hand, reinstall `georgia-legal-search@georgia-legal-search`, and test it in a new thread.
