# Georgian Legal RAG — Session Handoff (paste this into the next session)

> **This handoff is one focused task.** CLAUDE.md already points a fresh session at
> `memory-bank/` (architecture + blast-radius contracts) and `coordination/` (multi-session
> protocol) — skim those for context, then do the task below.
> _The full pre-2026-07-13 handoff (state per part, corpus numbers, gotchas) is preserved at
> `coordination/HANDOFF.snapshot-2026-07-13-before-reset.md` if you need it._

## Task — build `ingest/data/law_aliases.json` → A/B `CITATION_ROUTE=full`

**Current state**
- Corpus `georgian_legal` ≈ **2.65M points**; eval on **`golden_set_v2`** (337 pairs,
  `eval_set_hash 753e2985315be3e4`; known-item = legal_citation + keyword ≈ 120 queries).
- **Serving = best config LIVE:** `SEARCH_BACKEND=remote` → RunPod serverless endpoint
  **`ud8aetmo9aow1y`**, with **`CITATION_ROUTE=ids` + `RERANK_CANDIDATES=50`**, serving
  fingerprint **`5bdd6800701dbeb6`**. Confirm via `mcp__legal_rag__legal_health` after a `/mcp` reconnect.
- **Answer-quality eval subsystem (committed):** `ingest/eval/answer_eval.py`,
  `eval/eval_answer_quality.py`, `eval/diagnose_confident_wrong.py`, `eval/judge_eval.py`; 28 offline tests.

**Why this task (the #2 finding, settled 2026-07-13)**
Known-item **`confident_wrong` = 0.392** (rerank+citation): the reranker confidently (top score ≥ 0.92)
returns the WRONG law on ~2-in-5 known-item queries. `eval/diagnose_confident_wrong.py` on the 48 real
failures proved a **deterministic document_number guard catches only 1/48 (2%) = theater — do NOT build
it** (the advisory title/number + abstention safeguard is already live in the `legal_search` docstring,
`ingest/ingest/mcp_server.py` :425/:433, and is the right guard for the 85% "does this doc answer the
ask?" cases). The residual is a **retrieval-quality** problem. The one **untested** lever for the ~15
named-law legal_citation misses ("Civil Code art. 829", "Law on Entrepreneurs 189.5.d") is
**`CITATION_ROUTE=full`** (alias/title routing) — but it resolves NOTHING today because
**`ingest/data/law_aliases.json` was never built** (the diagnostic shows `full_route_resolvable = 0/48`,
entirely because the alias table is absent).

**Steps**
1. **Build `ingest/data/law_aliases.json`** — the alias table `extract_citation(mode="full")` consumes.
   Read `ingest/ingest/citations.py` (`_match_alias` / `load_aliases` / `_cached_aliases`) for the exact
   schema (`{"laws": [{"aliases": [...], "filters": {registration_code|document_number: ...}}]}`). Derive
   aliases from the most-cited base-law titles in the corpus (scroll Qdrant titles; matsne
   `is_consolidated=true` acts): map each well-known code/law name + common Georgian variants to its
   `registration_code`/`document_number`. Keep it **high-precision** — a loose alias mis-routes queries.
2. **A/B `CITATION_ROUTE=full` vs the live `ids`** on golden_set_v2: hybrid first (cheap, ~2 min,
   `RERANK_ENABLED=false`), then confirm under **rerank@50** (production mode):
   `python -m eval.eval_answer_quality --backend qdrant --mode rerank --golden-set v2 --citation-route full
   --translate-queries eval/query_translations_v2.json` for answer metrics; `eval/evaluate.py` for IR.
3. **Re-measure** with `python -m eval.diagnose_confident_wrong --run <new_run.json> --golden-set v2` —
   confirm the named-law failures move `pure_semantic → full_route_resolvable` and legal_citation
   identity@1 rises without new mis-routes.
4. **Gate (improvement.md §2, G1–G5):** legal_citation slice nDCG/R@10 **+≥0.03**, **no slice −0.02**
   (guards paraphrase / natural_question against alias mis-matches); `pytest tests -q` + `ruff check .`
   clean; paired `--compare` for a p-value.
5. **KEEP only on a gated win:** commit via **explicit pathspec** + log to `experiments.jsonl`; flip the
   worker knob (`CITATION_ROUTE=full` in the RunPod dashboard — browser tool) **only with user sign-off**
   (it changes `retrieval_fingerprint` → re-baseline).

**Coordination + guardrails**
- `citations.py` (mode=full / `_match_alias`) is owned by **session-verify-i1** — coordinate before
  editing it; building the NEW `law_aliases.json` data file + running evals needs no `citations.py` change.
- Standing rules: never train/eval on `holdout_doc_ids_v2.json`; never edit existing tests; never
  `git pull`/merge `origin/dev` (divergent LangChain fork); verify RunPod `pods=[]` after any job;
  `/mcp` reconnect after `ingest/.env` edits (the server caches code+env at spawn).
- Reference committed diagnostic: `ingest/eval/diagnose_confident_wrong.py` (commit `84ba313`).
