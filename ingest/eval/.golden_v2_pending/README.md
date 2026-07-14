# golden_set_v3 — pending work (resume kit)

**Status 2026-07-11:** `golden_set_v2.jsonl` is FROZEN at **337 verified pairs** and is now the
DEFAULT gating yardstick (`eval_set_hash 753e2985315be3e4`; `--golden-set` defaults to v2). **Do NOT
append to golden_set_v2.jsonl** — that changes its frozen hash and corrupts the gate anchor. The
authored-but-unappended pairs below must land as a **v3 additive superset**: create
`golden_set_v3.jsonl` = v2 (byte-identical) + these new pairs, `holdout_doc_ids_v3.json`,
`query_translations_v3.json`, and add a `"v3"` entry to `EVAL_SETS` in `eval/goldset.py`
(roots = same v1 + v2-delta). Then the "append" steps below target the v3 files, not v2.

This dir holds the authored work so a future session can finish to ~500 **without re-authoring**
(the LLM-authored pairs are the irreplaceable part; candidate docs + pair_plan are regenerable from
`scripts/sample_goldset_candidates.py` seed 20260710).

This directory is untracked (not committed) — it is a scratch resume-kit, NOT part of the eval set.

## What's here
- `pair_plan.jsonl` — 397-row plan (q104–q500), **with doc-swaps already applied** for reworks.
- `candidates.jsonl` — 307 sampled gold docs (seed 20260710), post parity/quarantine drop.
- `batches/b{3,6,7,8}/` — `raw/*.json` (author outputs), `batch.jsonl` (assembled),
  `verdicts/*.json` (adversarial verdicts), `rework_assign.json`.
- `docs_bundles.tar.gz` — per-doc `{source}__{id}.body.txt` + `.meta.json` (extract to a scratch
  dir; authors/verifiers read these). Regenerable from `snapshots/v1` + `snapshots/v2-delta`.
- `assemble_batch.py` (computes offsets from quotes — **never trust agent offsets**),
  `add_holdout.py`, `prepare_authoring.py`, `style_samples.json`, `delta_ids.json`.

## Remaining work (needs fresh subagents — the session subagent limit stopped these)
| batch | state | to do |
|---|---|---|
| b3 rework | `raw/rework.json` = 15 pairs, machine-clean, **unverified** | adversarially verify → append accepts |
| b6 | 50 pairs assembled, machine-clean, **0 verdicts** | adversarially verify (5 slices) → rework rejects → append |
| b7 | 20 pairs authored (a2,a3); a1/a4/a5 never ran | author a1/a4/a5, assemble, validate, verify, append |
| b8 | 0 authored (only `assign_*.json`) | author all 5, assemble, validate, verify, append |

Reaching ~500 total / per-slice n≥80 needs all four. Current slice counts (in committed v2):
natural_question 75, legal_citation 67, cross_lingual 60, keyword 55, temporal 40, paraphrase 40.
Also re-author dropped **q295** (paraphrase — needs a document-specific factual span, not a shared
legal definition) and **q220** is folded into b3 rework.

## Exact resume recipe (per batch)
```bash
SCR=<a fresh scratch dir>;  tar -xzf docs_bundles.tar.gz -C $SCR   # gives $SCR/docs/
cd ingest
# 1. VERIFY (fresh subagents, NOT the author): feed batches/bN/verify_*.json (build slices with
#    _body_file/_meta_file pointing at $SCR/docs/); each pair → accept/reject with sibling-grep.
# 2. REWORK rejects (doc-swap sibling-ambiguity cases); re-verify.
# 3. ASSEMBLE + machine-validate:
python3 <pending>/assemble_batch.py <pending>/batches/bN     # computes offsets
python3 <pending>/add_holdout.py <pending>/batches/bN/batch.jsonl   # → holdout_doc_ids_v2.json
uv run python scripts/validate_golden_batch.py --batch <pending>/batches/bN/batch.jsonl \
    --eval-set v2 --qdrant --existing eval/golden_set_v1.jsonl eval/golden_set_v2.jsonl
# 4. APPEND accepted records to eval/golden_set_v2.jsonl, EN pairs to query_translations_v2.json.
# 5. Full-file invariants (reground/enforce_holdout/lint_span_coverage on v2), pytest, ruff, commit.
```
NOTE `assemble_batch.py` hard-codes the original SCRATCH path — edit the `S = Path(...)` line to
your new scratch dir, and it reads `pair_plan.jsonl` (copy it next to the script or fix the path).

## Hard rules (unchanged)
v1 files FROZEN · never append unverified pairs · queries pin the doc via own identifier / unique
entity / unique fact (the #1 rejection cause is sibling-doc non-discrimination — verifiers corpus-grep)
· paraphrase = zero content-word overlap AND document-specific span · PII stays verbatim · all local.

## THE #1 lesson for authors (bake into every author prompt)
A query answerable by many sibling docs (shared boilerplate / legal definition / template phrase)
gets REJECTED. Pin every query on the doc's own document_number/registration_code/case-number, or a
unique named entity / quantitative fact. First-pass rejection fell from ~30% (early batches) to 0–6%
once this was in the prompt.
