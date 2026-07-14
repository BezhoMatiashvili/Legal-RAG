# Georgian Legal RAG — Quality-Improvement Runbook (next-session plan)

> **Audience:** a future Claude Code session in this repo.
> **What this is:** the prioritized next-session plan for raising **retrieval quality**. It
> does NOT change code by itself — it tells you what to do, in what order, and how to gate it.
> **First read:** `HANDOFF.md` (state), `improvement.md` (gated queue + G1–G5 gates + ledger),
> `memory-bank/INDEX.md` + `contracts.md`, `coordination/README.md`.
> **Goal:** raise retrieval quality — overall nDCG@10 / Recall@10 **and** the citation +
> keyword slices specifically.
> **Budget: ≤ $15 GPU total** (RunPod balance ~$4.74 → top up as needed, never exceed $15).
> Gate every change on **golden_set_v2** via the `improvement.md` §2 loop.
> _Authored 2026-07-11 from a research pass (codebase audit + 2026 technique review)._

---

## Phase 0 — PROTECT WHAT WE HAVE (do this BEFORE any change)

The working tree holds a large amount of **uncommitted, multi-session work** and the live
index's freshest full backup lags the current state. Secure it first.

> **📦 Backups already taken (2026-07-11, corpus @ 2,654,818 pts) — where to find them:**
> `~/backups/` **and** mirror `~/backups-mirror/`. Each holds: the full corpus snapshot
> `georgian_legal-…-2026-07-11-14-14-31.snapshot` (~28.6 GB, tar-verified), `gls-git-*.bundle`
> (full history), `gls-uncommitted-*.tgz` (113 at-risk source files), `gls-extras-*.tgz` (v3
> pending pairs + patches). **Manifest + restore steps: `~/backups/BACKUPS.md`.**
> ⚠ Both copies are on the **same NVMe disk** (no external/second disk was mounted at backup
> time) — for true off-disk redundancy: `rsync -a ~/backups/ <external-or-remote>/`. Re-run the
> steps below to take a FRESH backup before any new risky change.

1. **Git-fork hazard (never violate).** Tree is on branch `dev` (custom stack).
   `origin/dev` is a **divergent LangChain fork** — **never `git pull`/`merge` it**, and
   **never `git reset --hard`** to anything you did not create (it clobbers
   `build_payload`/`is_consolidated`/offset chunker/I1). Recovery anchor = the latest
   custom-stack commit (`daa4386` as of 2026-07-11; verify with `git log`). See memory
   `[[git-branch-fork-hazard]]`.

2. **Back up the uncommitted working tree (fast, zero git-state change).** Do NOT tar the
   whole tree (the 27 GB `qdrant_storage`/`.state`, 8.5 GB `artifacts`, 6.5 GB `snapshots`
   make it huge and slow). Capture only the at-risk source + full history:
   ```bash
   mkdir -p ~/backups; TS=$(date -u +%Y%m%dT%H%MZ)
   git bundle create ~/backups/gls-git-$TS.bundle --all           # all committed history
   git ls-files -mo --exclude-standard -z | \
     tar czf ~/backups/gls-uncommitted-$TS.tgz --null --ignore-failed-read -T -   # uncommitted+untracked source
   tar czf ~/backups/gls-extras-$TS.tgz --ignore-failed-read \
     ingest/eval/.golden_v2_pending .improvements coordination   # untracked v3 pairs + patches
   ```
   Then **ask the user for permission to commit** the tree in logical chunks (safest durable
   protection). Some files belong to other/dormant sessions — per CLAUDE.md, do not revert or
   claim their work; coordinate before committing shared files.

3. **Take a FRESH full corpus snapshot before any re-embed/merge.** Snapshots are NOT
   bind-mounted (only `/qdrant/storage` is), so create + copy out of the container:
   ```bash
   cd ingest; KEY=$(grep '^QDRANT_API_KEY=' .env | cut -d= -f2)
   curl -s localhost:6333/collections/georgian_legal -H "api-key: $KEY" | \
     python3 -c "import sys,json;d=json.load(sys.stdin)['result'];print(d['status'],d['points_count'])"
   # require green + expected count, then:
   SNAP=$(curl -s -X POST "localhost:6333/collections/georgian_legal/snapshots?wait=true" \
     -H "api-key: $KEY" | python3 -c "import sys,json;print(json.load(sys.stdin)['result']['name'])")
   docker cp "legal-qdrant:/qdrant/snapshots/georgian_legal/$SNAP" ~/backups/ && tar -tf ~/backups/"$SNAP" >/dev/null && echo OK
   curl -s -X DELETE "localhost:6333/collections/georgian_legal/snapshots/$SNAP" -H "api-key: $KEY" >/dev/null  # reclaim container disk
   ```
   (`scripts/publish_snapshot.py --create` also works but writes local publish manifest state.)

4. **Untracked artifacts git won't protect:** `ingest/eval/.golden_v2_pending/` (~150 v3
   golden pairs), `ingest/ingest/citations.py` (I1), `ingest/.state/reembed_v2/rows/` (staged
   I6 rows — regenerable via `reembed_export.py`). Step 2 already captures the first two.

5. **Standing safety rules:** never delete/recreate the `georgian_legal` collection; a writer
   turns the index **yellow** (hybrid queries time out) — wait for green; **always**
   `runpod_rerank.py down` / verify `pods=[]` after any GPU job; the MCP server caches
   code/.env at spawn → `/mcp` reconnect after edits.

6. **Consolidation-flag hazard:** `is_consolidated` is payload-only and **overwritten by any
   re-embed** — after ANY re-embed/merge, **re-run `scripts/reconcile_consolidated.py`** or the
   "current law" filter silently degrades. (Relevant to P6.)

---

## 1 — Where quality stands (baseline — do not re-derive; re-baseline if the corpus changes)

- Corpus `georgian_legal` = **2,654,818 pts**, green, 0 missing.
- Serving: `SEARCH_BACKEND=remote` (RunPod serverless GPU), `RERANK_BACKEND=onnx` int8,
  `RERANK_CANDIDATES=50`. Fingerprint **`471ee93fc0199b03`**.
- Gate = **golden_set_v2** (337 pairs, frozen, `eval_set_hash 753e2985315be3e4`).
- **Best production baseline (rerank@50, translations-on, @2.65M pts, `ingest/.state/ref_v2.json`):
  overall nDCG@10 0.4504 / R@10 0.5401.**
- **Weakest slices (rerank@50):** paraphrase 0.131 (n=40, hard by design, directional),
  **keyword 0.411 (n=55 — weakest gate-able)**, cross_lingual 0.432 (n=60). citation 0.484,
  natural_question 0.484, temporal 0.732 (easiest).

---

## 2 — Priority queue (in order; focus = overall quality + citation/keyword; ≤ $15)

### P1 — Finish & gate I1 citation routing on v2  `[near-$0 · HIGHEST VALUE · ~90% built]`
**Why:** the single biggest lever on the board. A fresh uncommitted eval row
(`experiments.jsonl`, `citation_route:ids`, v2, translations-on) shows **legal_citation +0.100
nDCG / +0.090 R@10, keyword +0.043/+0.055, overall +0.041/+0.042, NO slice regresses** — a
clean pass of I1's original gate (legal_citation R@10 +≥0.10). It was blocked on v1 (gold docs
were amendments, not the numbered act; bare № ambiguous); v2's **67 citation pairs whose gold
IS the cited act** + the completed base-law corpus unblocked it.
**Where:** `ingest/ingest/citations.py` (`extract_citation`), `ingest/ingest/search.py:139`
(wire behind `CITATION_ROUTE`), `ingest/ingest/config.py:173` (fingerprint-folded when on);
patches in `.improvements/i1_citation_route_{full,partial}.patch`.
**⚠ Coordinate:** `citations.py` was being **actively edited 2026-07-11** — check
`coordination/sessions/` + the board before touching; pick it up only if abandoned.
**Do:** (a) confirm the win **under rerank@50** (production mode — the recall-stage win may be
absorbed or amplified by the reranker) — GPU rerank ~$0.30 (`RERANK_REMOTE_URL` +
`runpod_rerank.py up`, then `down`) or CPU ~2 h `nohup`; (b) run the **paired test**
`eval.evaluate --compare` for a p-value; (c) optionally build the `full` alias variant
(`ingest/ingest/data/law_aliases.json` — build from corpus titles, NOT golden queries); (d) if
it holds: commit + add an `improvement.md` §8 ledger row + flip the serving knob only with user
sign-off (it changes the fingerprint).
**Gate:** G1 legal_citation R@10 +≥0.10; G2 no monitored slice −0.02; G5 tests/ruff clean.

### P2 — Metadata auto-filtering / filter-prompting  `[near-$0 · docstring/serving · fingerprint-neutral]`
**Why:** all filter infra is built (`status`, `is_consolidated`, `date`, `document_type` in
`SearchInput` → `search.py:build_filter`) but fires **only when Claude-the-client chooses to
pass it** — no default precision filter for "current law" asks. Cheap precision lever for
citation/keyword/temporal.
**Where:** `ingest/ingest/mcp_server.py` `legal_search` docstring (follow the I4 pattern) — teach
the client: "for questions about *current* law, set `status=in_force` and/or
`is_consolidated=true`; for a specific act, `legal_lookup` by number first." Optionally a
serving-side intent→filter default (bigger change; docstring first).
**Gate:** G5 only (retrieval byte-identical unless a serving default is added) + a manual smoke
on 5 "current law" queries. Filters are NOT in the fingerprint.

### P3 — Grow the golden set to v3  `[near-$0 · enables everything downstream]`
**Why:** unlocks gating small wins and raises paraphrase/temporal above n≥80 (currently
"directional only"), AND provides training/hard-neg data for P5. ~150 authored pairs already
staged in `ingest/eval/.golden_v2_pending/` (see its `README.md`).
**Do:** land them as a **v3 additive superset** (new `EVAL_SETS` entry) — **never mutate the
frozen v2**. Follow the I5 recipe: span-anchored, `reground`-valid, holdout updated, loader
invariants + `tests/test_goldset*.py` green. Report the new distribution.
**Gate:** loader invariants pass; distribution reported; v2 stays the frozen anchor until user
signs off on gating v3.

### P4 — Stronger reranker A/B on the GPU worker  `[~$1–5 · no re-embed · biggest cheap NEW lever]`
**Why:** rerank is "the single biggest precision lever" and the serverless GPU makes a
bigger/newer model affordable. A better reranker lifts ALL slices — especially **paraphrase**
(pure-semantic) and cross_lingual. **CAVEAT (critical):** 2026 reranker gains are benchmarked on
English/Chinese; **none benchmark Georgian** — so this MUST be A/B'd on golden_set_v2, not
assumed. (Research: https://futureagi.com/blog/best-rerankers-for-rag-2026/ )
**Candidates (open, Apache-2.0, GPU-affordable):**
- **Qwen3-Reranker-0.6B / 4B** — cross-encoder emitting pairwise 0–1 scores (yes/no logits),
  100+ langs. Needs a small **adapter class** (mirror `ONNXBGEReranker` in
  `ingest/ingest/rerank.py`; `BGEReranker` hardcodes `_MAX_LENGTH=512` + `sigmoid` for a single
  classification head, so Qwen3/mxbai are NOT pure `RERANK_MODEL` swaps).
  https://huggingface.co/Qwen/Qwen3-Reranker-0.6B
- **mxbai-rerank-large-v2** (1.5B, Apache-2.0, 100+ langs) — same adapter caveat.
- **fine-tuned bge-reranker-v2-m3 or bge-reranker-large** — SAME head → **pure config drop-in**
  via `RERANK_MODEL` (`config.py:125`). Cheapest to try.
- (Avoid jina-reranker-v3: CC-BY-NC license + listwise API, doesn't fit our pairwise interface.)
**Where:** `ingest/ingest/rerank.py` (`make_reranker`, add adapter), `RERANK_MODEL` env; deploy on
the serverless worker (endpoint env `RERANK_MODEL` + image redeploy; model caches in the volume).
Keep the score-parity contract (`memory-bank/areas/retrieval-serving.md`).
**Gate:** golden_set_v2 rerank@50 — KEEP only if overall nDCG@10 improves ≥0.01 (or a
citation/keyword slice +≥0.03) with NO slice −0.02; re-calibrate the 0.92 abstention gate
(`scripts/calibrate_min_score.py`) for the new model; re-baseline (rerank_model is in the
fingerprint). Start with the pure-config bge variants (cheapest), then one adapter model.

### P5 — In-domain reranker fine-tune  `[~$2–5 · no re-embed · most TARGETED for Georgian legal]`
**Why:** newly unlocked by I5 (v2 golden set). Generic rerankers lack Georgian-legal specificity;
fine-tuning a cross-encoder on in-domain pairs targets exactly that. Sentence-Transformers has a
mature CrossEncoder trainer + hard-negative mining
(https://huggingface.co/blog/train-reranker); `[query, positive, negative]` format is best. A
fine-tuned **bge-reranker-v2-m3** stays same-head → pure `RERANK_MODEL` drop-in.
**⚠ THE TRAP — eval contamination:** the 337 golden_set_v2 pairs are the TEST set — **never train
on them.** Build the train set from the **v3-pending pairs (P3)** + `.state/queries.jsonl` logged
queries + mined hard negatives, and hold out golden_set_v2 entirely (respect
`holdout_doc_ids_v2.json`). No fine-tuning infra exists yet → build a small
`scripts/finetune_reranker.py` (new; add a test). GPU fine-tune of a 568M cross-encoder on a few
hundred–thousand pairs is minutes and cheap.
**Gate:** golden_set_v2 rerank@50, fine-tuned vs base; KEEP iff overall nDCG +≥0.01 and no slice
−0.02; re-baseline + re-calibrate abstention. Do AFTER P3 (needs the train data) and ideally on
the P4 winner as the base.

### P6 — Relaunch the shelved I6 contextual-headers re-embed  `[~$4 · full re-embed · if 4090 capacity]`
**Why:** Anthropic Contextual-Retrieval-style lift on boilerplate-heavy amendment acts
(`improvement.md:279`). Fully staged: 14 shards / 2,654,818 rows exported
(`.state/reembed_v2/rows/`), gate ref `ref_v2.json`, batch-size bug fixed, one-command relaunch in
`i6_post_scrape_runbook.md`. Re-embeds into `georgian_legal_v2` (identical ids/payloads →
cleanest A/B).
**Blocker:** RunPod 4090/multi-GPU `SUPPLY_CONSTRAINT` (only A5000 → ~17 h, doesn't fit) + budget.
Relaunch when 4090 capacity returns; the orchestrator auto-gates (rerank nDCG ≥ ref +0.02, no
slice −0.02), pulls+restores only on PASS, always terminates.
**After it lands:** re-run `reconcile_consolidated.py` (Phase 0 §6), re-baseline everything, flip
`COLLECTION_NAME`/`EMBED_HEADER_V2` only on a gated PASS + user sign-off.
**Gate:** overall nDCG@10 +≥0.02 (orchestrator-enforced).

---

## 3 — Explicit DO-NOT (measured dead-ends — `improvement.md:311`; do not re-propose)
MMR diversity (measured harmful 0.289→0.074) · blanket multi-query / RAG-fusion (reformulation
belongs in the agentic loop, miss-triggered) · HyDE-by-default (must be Georgian, gated only) ·
ColBERT/multivector (same slot as the cross-encoder, ~hundreds of GB) · rerank depth >80 (gains
saturate) · **full embedding-model swap / hybrid re-architecture** (too big for the ≤$15 budget +
loses BGE-M3 learned-sparse — out of scope) · editing the frozen golden set · committing without
asking.

## 4 — Gating & verification protocol
Run the `improvement.md` §2 loop for EVERY item: one change at a time → tests+ruff clean → measure
(cheap hybrid ~2 min for recall-stage; confirm under rerank@50) → **gates G1–G5** (G1
target-metric threshold, G2 no slice −0.02, G3 wins <0.02 need paired p<0.05, G4 latency, G5
tests+fingerprint) → KEEP (commit + ledger row) or REVERT (record anyway). Always `--log` to
`experiments.jsonl`. **Re-baseline rule:** never compare across corpus states; record
`points_count` per row.

## 5 — Budget ledger (≤ $15; balance ~$4.74 → top up)
Near-$0: P1 confirm (CPU) · P2 · P3. GPU spend: P1 rerank confirm (~$0.30) · P4 A/B (~$1–3) · P5
fine-tune (~$2–4) · P6 re-embed (~$4). Sequence P1→P2→P3 first (free), then spend on P4/P5
(highest ROI, no re-embed), keep P6 last (needs 4090 + biggest spend). Log every GPU $;
`runpod_rerank.py down` + verify `pods=[]` after each.

## 6 — Coordination & cross-refs
Register in `coordination/sessions/`, claim files, watch the board + `git status --short`
(multi-session repo). This runbook complements `improvement.md` (the gated queue — add ledger rows
there) and `HANDOFF.md` (state). Update memory-bank `areas/` + `contracts.md` in the same session
for any touched symbol; `python3 ingest/scripts/gen_code_map.py --check`.
