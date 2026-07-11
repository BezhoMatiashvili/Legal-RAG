# improvement.md — Retrieval Accuracy Runbook (for Claude Code)

**Audience:** a Claude Code session working in this repo.
**Goal:** raise retrieval accuracy of the Georgian legal RAG toward "always finds the right document",
one gated improvement at a time. **Every attempt is measured; it is KEPT only if it passes the gate
in §2, otherwise it is REVERTED.** No exceptions, no "it probably helps".

**Out of scope:** corpus backfill of consolidated base laws — **the user handles that**
(`scripts/backfill_consolidation.py`). But see §8 (re-baseline rule) and I6 (piggyback on the
re-embed) because that work interacts with this runbook.

---

## 0. Hard rules (read before touching anything)

1. Read `HANDOFF.md` and memory first. Current recommended config: **`hybrid+rerank@50`,
   `retrieval_fingerprint=81c807b279399098`** (`ingest/eval/phase_c_report.md`).
2. **Git hazard:** local `dev` is a custom stack; `origin/dev` is a divergent LangChain fork.
   **Never `git pull`/merge origin/dev. Never `git reset --hard` to anything you didn't create.**
3. **Do not commit without asking** — but this runbook *requires* commits for safe revert, so
   **Step 0 of the loop is asking the user for checkpoint-commit permission** (§2).
4. **Never edit existing tests to make them pass.** Fix code. If data looks wrong, tell the user.
5. **No external APIs anywhere in ingest/embed/eval.** Anything the eval harness consumes must be a
   static, checked-in artifact (e.g. a translations JSON that *you* author in-session — that is
   allowed; a runtime API call from harness code is not). PII stays in the index.
6. **RAM (30 GB box):** never load the reranker for non-rerank eval modes — export
   `RERANK_ENABLED=false` for those runs (already baked into `ingest/scripts/phase_c_full.sh`).
   Never run `--mode bm25` via the in-memory scroll path; the disk index `eval/.bm25_full/` exists.
7. **MCP staleness:** after editing anything under `ingest/`, the running `legal_rag` MCP server is
   stale — tell the user to reconnect via `/mcp` before judging behavior through MCP tools.
   The direct CLI (`.venv/bin/python -m ingest search "<q>" --top-k 5`) always uses current code.
8. **RunPod:** only for explicitly approved batch jobs. Balance is ~$4.30. If a pod is ever
   started: **always** `python scripts/runpod_rerank.py down` after, and verify `pods=[]`.

---

## 1. Preflight (before any improvement session)

Run from `ingest/` unless noted.

```bash
# 1) Qdrant up + collection green (it is currently NOT running by default)
docker compose up -d
curl -s localhost:6333/collections/georgian_legal | python3 -m json.tool | grep -E 'points_count|status'
# Require: status "green". If yellow → an ingest/backfill is running → STOP and wait.

# 2) Record corpus state for the baseline ledger (§8): note points_count.

# 3) Tests + lint must be clean BEFORE you change anything
.venv/bin/python -m pytest tests -q     # 222 pass as of 2026-07-09
.venv/bin/python -m ruff check .
```

**Baseline:** `ingest/eval/experiments.jsonl` holds Phase C results, but if `points_count` differs
from the ledger (§8) — e.g. the user's consolidation backfill landed — **re-run the baseline before
comparing anything**:

```bash
# fast recall-stage baseline (~2 min)
RERANK_ENABLED=false .venv/bin/python -m eval.evaluate --backend qdrant --mode hybrid --relevance chunk --log
# full-quality baseline (CPU rerank@50 ≈ 67 s/query ≈ 2 h for 103 queries → run in background)
nohup .venv/bin/python -m eval.evaluate --backend qdrant --mode rerank --rerank-candidates 50 --relevance chunk --log \
  > /tmp/eval_rerank_baseline.log 2>&1 &
```

Reference numbers (2026-07-09, pre-consolidation corpus, chunk relevance, k=10):
bm25 nDCG 0.192 / hybrid 0.182 / **hybrid+rerank@80 nDCG 0.289, R@10 0.388**; cross-lingual slice
(22 EN→KA pairs): bm25+sparse 0.000, hybrid 0.136, rerank@80 R@10 0.273.

---

## 2. The Improvement Loop (run this for EVERY attempt)

**Step 0 — checkpoint permission (once per session).** Ask the user:
*"May I make a checkpoint commit of the current tree on `dev`, and one commit per kept
improvement, so failed attempts can be reverted cleanly?"* If **yes**: `git add -A && git commit`
(checkpoint). If **no**: before each attempt save `git diff HEAD > .improvements/attempt_<n>.patch`
plus the list of new files, and revert by `git apply -R` + deleting the new files. The commit path
is strongly preferred — the tree currently carries a lot of uncommitted work and a botched manual
revert would destroy it.

**Step 1 — implement ONE improvement** from the queue (§3), smallest possible diff. New retrieval
behavior must be a **switchable knob** (env var or CLI flag folded into `config_hash`, following the
existing pattern in `eval/evaluate.py` + `eval/backend.py` `knobs`), so ON vs OFF is measurable and
revert is trivial.

**Step 2 — tests.** `.venv/bin/python -m pytest tests -q` and `ruff check .` must be fully clean.
Add new tests for new code (never modify existing ones). Any failure → fix or revert; do not proceed.

**Step 3 — measure.** Iterate cheap, confirm expensive:
- **Iterate** with a non-rerank mode (`hybrid`, ~2 min/run, `RERANK_ENABLED=false`) — every
  recall-stage change (routing, translation, fusion weights) is visible here.
- **Confirm** the surviving candidate with one background rerank@50 run (~2 h) — the production mode.
- Where the harness supports it, use the **paired test**:
  `--compare <baseline_mode> <candidate_mode>` (paired permutation p-values).
- Every run uses `--log` (appends to `eval/experiments.jsonl` — the permanent record).

**Step 4 — gate.** KEEP the change **only if ALL of these hold**:

| # | Condition |
|---|-----------|
| G1 | The improvement's **target metric** (stated per-item in §3) improves by at least its stated threshold. |
| G2 | **No monitored metric regresses by > 0.02 absolute**: overall `ndcg10` and `recall10`, `per_language.ka`, `per_language.en`, and all five `per_query_type` slices. (0.02 tolerance = bootstrap noise floor at n=103; anything worse is a real regression.) |
| G3 | Point-estimate wins **< 0.02 count as indeterminate → do NOT keep** (unless the paired `--compare` p < 0.05). Honesty over motion. |
| G4 | Latency: rerank-mode `latency_ms.total.p50` does not grow > 10% vs baseline, and non-rerank p50 stays < 1 s. |
| G5 | Tests + ruff clean; `retrieval_fingerprint` changes **only if** the change is intentionally part of the serving config. |

**Step 5 — keep or revert.**
- **KEEP:** commit as one commit (`improvement: <name> — nDCG@10 X→Y, slices …`); update the
  ledger (§8) and `HANDOFF.md`.
- **REVERT:** `git checkout -- <files>` + delete new files (safe only on top of the checkpoint
  commit), or `git apply -R` the patch. Record the attempt in the ledger anyway with its numbers
  and a one-line cause — **failed experiments are data**; the `experiments.jsonl` rows stay.

**Step 6 — next item.** One improvement at a time; never stack two unmeasured changes.

⚠ **No peeking / no re-rolls:** decide the gate thresholds *before* the run (they are fixed above);
do not re-run an eval hoping for a better draw, and do not tweak-and-retest more than twice per
item — after two failed variants, mark the item BLOCKED in the ledger and move on.

---

## 3. Improvement queue (in order — evidence × cost)

### I1 — Citation & alias exact-match routing  `[recall-stage, eval-measurable]`

**Why:** embeddings structurally fail referential queries ("law №432", "შრომის კოდექსი მუხლი 31");
adding an exact/label path fixed +30–41% similarity on statute-reference queries (Poly-Vector
Retrieval, arXiv 2504.10508); ~20% of real legal-search sessions are known-item lookups. The
infrastructure half-exists: `legal_lookup` (`ingest/ingest/mcp_server.py:674`) does exact
`document_number`/`registration_code` filtering, and payload keyword indexes for both already exist
(`qdrant_store.ensure_collection`). What's missing is (a) detection inside the search path and
(b) an alias table.

**Implement:**
1. New module `ingest/ingest/citations.py`: `extract_citation(query) -> CitationRef | None` —
   regexes for matsne document numbers (`№?\s?\d+(-\S+)?`) with legal context words, registration
   codes (`\d{3}\.\d{3}\.\d{3}…` patterns — check real values in payloads first), and quoted/known
   law titles.
2. Alias table `ingest/ingest/data/law_aliases.json`: canonical law → `{registration_code,
   document_number, aliases: [ka official, ka informal, en, ru, translit]}`. **Build it from corpus
   titles** (scroll distinct matsne titles by frequency, take top ~100 laws + all codes), NOT from
   golden-set queries (that would be overfitting the eval).
3. Wire into `hybrid_search` (`ingest/ingest/search.py:102`) behind a knob (env
   `CITATION_ROUTE`, default off): if a citation/alias resolves → run the exact `build_filter`
   lookup; if it returns hits, **prepend** them (score-pinned above semantic hits) to the semantic
   results; if not, fall through to normal search unchanged.
4. Eval knob: `--citation-route` flag in `eval/evaluate.py` → `QdrantBackend`, folded into
   `config_hash` `knobs` (copy the pattern of `--sparse-weight`).

**Measure:** hybrid mode, ON vs OFF, then `--compare`. Target slice: `per_query_type.legal_citation`
(17 pairs) and `keyword` (21 pairs).
**Gate:** G1 = `legal_citation` R@10 +≥0.10 (≥2 extra pairs hit). G2–G5 standard — by construction
non-citation queries must be byte-identical (assert in a test: query without citation ⇒ same
results as knob-off).
**Revert:** knob off + revert commit.

### I2 — Georgian query translation for cross-lingual  `[recall-stage, eval-measurable]`

**Why:** strongest evidence found for this system: BGE-M3 has a measured native-language retrieval
bias, and **Georgian is explicitly listed among languages that "only retrieve native documents"**
(BordIRLines, arXiv 2410.01171). Our EN slice confirms it: sparse 0.000, hybrid 0.136 nDCG. Routing
(dropping sparse for EN) was a statistical tie — translation attacks the actual cause. Zero risk to
KA queries (fires only on non-KA input, which `detect_language` at `search.py:16` already detects).

**Implement (two layers):**
1. **Eval layer (measurable):** author `ingest/eval/query_translations_v1.json` — high-quality
   Georgian legal-register translations of the 22 EN golden queries, keyed by golden-set `id`.
   *You (Claude Code) write these in-session* — you are the translator; no runtime API calls.
   Add backend knob `--translate-queries <path>`: for queries with an entry, embed the Georgian
   text instead (variant A), or RRF-merge results of original+translated (variant B — try A first;
   B costs a second search per query). Fold into `config_hash`.
2. **Serving layer:** `legal_search`'s docstring (`mcp_server.py:314`) gets one added paragraph:
   *"The corpus is Georgian. For non-Georgian queries, translate the query into Georgian legal
   terminology and search with the Georgian text (keep the original only as a fallback);
   Georgian-language statute vocabulary retrieves dramatically better."* Claude (the MCP client)
   does the translating at answer time — consistent with the MCP-first architecture.

**Measure:** hybrid mode ON vs OFF; target slice `per_query_type.cross_lingual` /
`per_language.en`; confirm with rerank@50.
**Gate:** G1 = cross-lingual R@10 +≥0.09 (≥2 of 22 pairs). KA slices mechanically unchanged —
assert that in a test.
**Revert:** knob off; docstring paragraph is kept only if the eval knob won (they stand together).

### I3 — Fusion & recall tuning sweep (finish the deferred grid, then weighted fusion)  `[config-only]`

**Why:** this is the Phase C sweep that was deferred (HANDOFF §deferred). Additionally, published
results say tuned weighted fusion beats plain RRF by ~3–8% relative (Bruch et al., TOIS; OpenSearch
−3.86% nDCG for RRF), and BGE-M3's own paper weights sparse at **0.3** (dense 1.0) — we currently
fuse unweighted. The harness already has every knob: `--fusion {rrf,dbsf}`, `--sparse-weight`,
`--prefetch-limit`, `--hnsw-ef`, `--rescore`.

**Implement:** nothing new first — run the existing grid on a **green** index:
```bash
nohup bash scripts/phase_c_full.sh tuning > /tmp/tuning.log 2>&1 &
```
Then targeted: `--sparse-weight {0.2,0.3,0.5,0.7,1.0}` on hybrid mode, reading overall + `en` + `ka`
slices. If the optimum clearly differs by language (expect: near-0 for EN, ~0.3–1.0 for KA), add a
per-language sparse weight to the routing branch in `search.py` behind a knob.

**Measure/Gate:** winner vs current hybrid via `--compare`; G1 = overall nDCG@10 +≥0.02 (or a
cross-lingual win ≥0.09 with G2 intact). Prefetch/ef/rescore changes gate on **latency at equal
quality** instead. Confirm final winner under rerank@50 (fusion feeds the reranker's candidates —
better recall@50 must show up there, else it doesn't matter).
**Revert:** these are config values — revert = restore previous values.

### I4 — Agentic retrieval playbook + abstention + citation existence-check  `[MCP layer]`

**Why:** iterative retrieval is worth +5 to +21 recall points over single-shot in published agentic
RAG evals (IRCoT, Self-Ask, Search-o1) and it is pure prompting — our client IS Claude. Unprompted
abstention is known-bad (~25% refusal when it should refuse); a prompted sufficiency check reaches
~93% (Google "Sufficient Context", ICLR 2025). Citation existence-checking achieved **zero
hallucinated citations** by construction in a legal RAG study. Tool descriptions are rich today but
contain **zero retry/abstention guidance** (verified).

**Implement (docstrings + one script; no retrieval-code change):**
1. `legal_search` docstring — add a **retry playbook**: if results look wrong → (a) re-query in
   Georgian legal terminology; (b) strip filters; (c) if the ask names a law/number → `legal_lookup`
   first, then `legal_get_document`; (d) browse by `registration_code` via
   `legal_get_document_versions` for lineage; (e) raise `top_k`. Think between calls; stop when the
   returned `document_number`/`title` actually matches the ask.
2. **Abstention contract** in the docstring: scores are calibrated 0–1; *"if the top score is below
   ~0.45 or no hit's title/number matches the asked-for law, report that the document was not
   found rather than citing the nearest match — the corpus may lack it."*
   Calibrate the number first: new `scripts/calibrate_min_score.py` runs the 103 golden queries,
   collects top-1 reranker scores for gold-hits vs non-hits, prints the distribution; pick the
   threshold that keeps ≥ 99% of gold hits. Also decide whether serving `RERANK_MIN_SCORE`
   (currently 0.3) should move — same data.
3. **Citation existence-check** paragraph in `legal_search` + `legal_get_document` docstrings:
   *"before emitting any citation in an answer, verify it resolves via `legal_lookup`
   (document_number or registration_code); never cite an unresolvable identifier."*

**Measure:** the retrieval harness cannot see docstrings — the gate is different here:
tests pass (add tests asserting the docstrings contain the playbook/abstention markers — cheap
drift guards), retrieval metrics **byte-identical** (no code path touched; verify fingerprint
unchanged), plus a manual smoke: 5 scripted hard queries through the direct CLI and through MCP
(after `/mcp` reconnect) checking the behavior contract reads correctly in tool output.
**Gate:** G5 only + zero retrieval diffs. **Revert:** revert commit.

### I5 — Grow the golden set to 150–250 pairs  `[measurement power]`

**Status (2026-07-11): LANDED (partial) — `golden_set_v2.jsonl` at 337 verified pairs** (103 frozen v1
byte-identical + 234 new). Harness plumbing complete and committed: `--golden-set {v1,v2}` flag,
multi-root `SnapshotBodies` (v1 + additive `snapshots/v2-delta/`), `scripts/build_snapshot_delta.py`,
`scripts/validate_golden_batch.py`, `scripts/sample_goldset_candidates.py` (seed 20260710), 34 new
tests. New `temporal` query_type added (breakdown() is data-driven). Acceptance eval logged (v2 hybrid
nDCG@10 0.385 / R@10 0.528 @ 2,638,482 pts; v1 consistency reproduces recall10 0.359, eval_set_hash
frozen). **Gating STAYS on v1 until user sign-off.** Remaining to hit 500 / per-slice n≥80: verify+append
batches 6 (50, machine-clean) + b3-reworks (15) + author b7/b8 (~97) — authored/pending work preserved in
`ingest/eval/.golden_v2_pending/`; the session subagent limit stopped the last three batches. This
unblocks I1 (citation slice now has 67 pairs whose gold IS the cited act, `is_consolidated` base laws).

**Why:** at n=103, IR-eval statistics (Webber/Moffat/Zobel: ≥150 topics; ~164–262 topics to detect
δ≈0.033) say we cannot certify small wins — G3 above exists because of this. Every later
improvement gets sharper the moment this lands.

**Implement:** **additive only.** Create `ingest/eval/golden_set_v2.jsonl` = v1 + new pairs
(v1 stays frozen — its `eval_set_hash` anchors all history). New pairs follow the exact v1 recipe:
span-anchored `relevance[]` with `char_start/char_end` into snapshot `body_markdown`,
`evidence_quote` byte-exact (the loader's `reground` fails loud otherwise), gold docs added to
`holdout_doc_ids.json`. Priorities: more `cross_lingual` (22→~50) and `legal_citation` (17→~40)
pairs — the two slices this runbook's improvements target; sample target docs from corpus strata
(source × document_type × date), **not** from documents the improvements were tuned on.
Generate → span-validate → adversarially re-verify each pair against the full document text
(v1's process). Wire `--golden-set <path>` selection if not already parameterized.

**Gate:** loader invariants pass (reground, holdout, span-coverage lint), `pytest tests/test_goldset.py`
green, distribution table reported to the user. From then on: **gate decisions on v1 (frozen
yardstick), report v2 alongside** — switch gating to v2 only after the user signs off.
**Revert:** delete the v2 file (v1 untouched by construction).

### I6 — Contextual embed-headers v2  `[requires re-embed — COORDINATE WITH USER]`

**Why:** Anthropic Contextual Retrieval: −35–49% retrieval failures from prepending document context
to chunks before embedding; a legal-specific study halved doc-level mismatch on boilerplate-heavy
corpora (amendment acts!). We already prepend `title > document_type > heading_path`
(`chunking.build_embed_text:279`) — the upgrade is adding `document_number`, `date`, `status`
(`in_force`/`repealed`), and consolidation marker to the embedded prefix (stored text stays clean).

**Implement only when the user schedules the next full re-embed** (likely after the consolidation
backfill — one RunPod job, ~$4 at 8×4090 ≈ 40 min, balance is ~$4.30, so it must be combined, not
separate). Change `build_embed_text`, keep a knob to reproduce v1 text for A/B. **Everything
re-baselines after** (new vectors = new world, §8).
**Gate:** overall nDCG@10 +≥0.02 on the post-re-embed baseline pair (old-header vs new-header would
need two embeds — if budget forbids, gate v2-headers against the fresh baseline and accept the
confound, stating it in the ledger).
**Status: PARKED until user green-lights the re-embed.**

### I7 — ONNX int8 reranker  `[latency, not accuracy]`

**Why:** CPU rerank@50 ≈ 67 s/query is the serving bottleneck; int8 dynamic quantization ≈ 2×+
speedup at <1 nDCG point loss (sentence-transformers documents the export for this exact model).
Faster rerank also makes future rerank-mode evals ~2× cheaper — do this earlier if eval turnaround
becomes the constraint.

**Implement:** export via `sentence-transformers` cross-encoder ONNX path; new
`ONNXBGEReranker` in `ingest/ingest/rerank.py` behind `RERANK_BACKEND=onnx` (default unchanged).
**Measure:** full golden set rerank@50, int8 vs fp32.
**Gate:** nDCG@10 within 0.01 of fp32 AND p50 rerank latency ≤ 0.5× fp32. **Revert:** knob off.

---

## 4. Explicit DO-NOT list (measured or evidence-based)

- **MMR diversity** — measured **harmful** here (nDCG 0.289 → 0.074). Off, stays off.
- **Blanket multi-query / RAG-fusion** — not significant after reranking in deployment studies;
  latency cost is real. Reformulation belongs in the agentic loop (I4), triggered on miss only.
- **HyDE by default** — net-negative pre-rerank in domain studies; known to fail for low-resource
  languages. Only ever as a gated A/B, and the hypothetical text must be **Georgian**.
- **ColBERT/multivector** — ~+1 nDCG on top of dense+sparse, same slot as our cross-encoder,
  ~hundreds of GB storage at 2.45M chunks. No.
- **Rerank depth > 80** — published depth ablations: gains saturate; some combos degrade past peak.
- **Embedding fine-tuning** — decided against (cost, cross-lingual forgetting risk, eval-set
  contamination). Revisit only after I5 provides a train/test split and the user re-opens it.
- `--mode all` / in-memory bm25 on the 2.45M-chunk collection; evals against a **yellow** index;
  leaving RunPod pods running; editing existing tests; committing without permission.

---

## 5. Metrics glossary (what the gate reads)

Each `--log` row in `eval/experiments.jsonl` has: `metrics` (`recall5`, `recall10`, `ndcg10`,
`mrr10`), `cis` (bootstrap 95%), `per_query_type` (natural_question 32 / cross_lingual 22 /
keyword 21 / legal_citation 17 / paraphrase 11), `per_language` (ka 81 / en 22), `latency_ms`
(embed/search/rerank/total p50/p95), `config_hash`, `knobs`. Primary = `ndcg10` @ `--relevance
chunk`. `config_hash` identifies the eval config; `retrieval_fingerprint` identifies the serving
config — don't conflate them.

## 6. Runtime budget cheat-sheet

| Run | Cost |
|---|---|
| hybrid/dense/sparse eval (103 q, `RERANK_ENABLED=false`) | ~2 min |
| rerank@50 eval, CPU | ~2 h (background, `nohup`) |
| rerank@80 eval, CPU | ~3.3 h |
| rerank eval on RunPod GPU (`RERANK_REMOTE_URL` + `scripts/runpod_rerank.py`) | ~10 min, ~$0.30 — **fragile, always `down` after; ask user first (balance $4.30)** |
| full tuning grid `phase_c_full.sh tuning` | hours; background overnight |

## 7. Session end checklist

- [ ] All kept changes committed (one commit each); nothing half-applied in the tree.
- [ ] Failed attempts reverted AND recorded in the ledger below.
- [ ] `pytest` + `ruff` clean.
- [ ] `HANDOFF.md` "NEXT SESSION" updated; memory updated if a standing fact changed.
- [ ] No RunPod pods running (`pods=[]`).

---

## 8. Ledger (append one row per attempt — kept or not)

**Re-baseline rule:** eval numbers are comparable **only at the same corpus state**. Record
`points_count` with every row; when it changes (user's consolidation backfill, re-embeds), re-run
the §1 baselines before the next gate decision. Never compare across corpus states.

| Date | Item | Corpus points | Mode/knobs | Target metric Δ | Slice check (G2) | p (if paired) | Decision | Commit / cause |
|---|---|---|---|---|---|---|---|---|
| 2026-07-09 | (baseline, Phase C) | 2,453,915 | hybrid+rerank@80 | nDCG@10 0.289, R@10 0.388 | XL R@10 0.273 | — | baseline | see phase_c_report.md |
| 2026-07-10 | (re-baseline, post-delta-merge) | 2,637,645 | hybrid, no knobs | nDCG@10 0.171, R@10 0.311 | XL R@10 0.136 · citation R@10 0.529 · paraphrase 0.000 | — | baseline | corpus grew +183k pts; all gates now compare at this state |
| 2026-07-10 | I1 citation routing (variant `ids`) | 2,637,645 | hybrid `--citation-route ids --ab` | citation R@10 Δ **+0.000** (gate ≥+0.10); R@5 −0.118 | overall TIE (p≥0.40); XL R@5 +0.045 | 1.00/0.55 | **REVERTED → BLOCKED** | structural mismatch: golden citation queries' gold docs are *decisions/amendments citing* the number, not the numbered act (which routing retrieves — and base acts are largely absent pre-consolidation); bare № numbers are massively ambiguous in matsne (№71→municipal budgets). 4/7 firings clean-fall-through, 3/7 wrong-doc pins. Variant `full` (aliases) inherits the same mismatch → not spent. Re-attempt only after consolidation backfill + I5 adds known-item pairs whose gold IS the cited act. Patch: `.improvements/i1_citation_route_full.patch` |
| 2026-07-10 | **I2 EN→KA query translation (variant A)** | 2,637,645 | hybrid + rerank@50 `--translate-queries eval/query_translations_v1.json --ab` (file hash `ac49dd4a013ae560`) | **cross_lingual R@10: hybrid +0.227 (0.136→0.364), rerank@50 +0.227 (0.273→0.500)** — gate ≥+0.09 | overall nDCG@10 hybrid +0.043 (p=0.010 ADOPT), rerank@50 0.286→0.320; keyword/citation/natural_question byte-identical; ⚠ paraphrase 0.091→0.000 in the GPU pair — NOT the knob (all-KA slice, translations file has only the 22 EN ids; unit test asserts identical KA requests) — index/optimizer noise, same KA jitter seen between arms of every run today | 0.010 (hybrid nDCG) | **KEPT** | eval knob + `legal_search` docstring guidance ("translate non-Georgian queries to Georgian legal terminology"); serving fingerprint unchanged |
| 2026-07-10 | I3 fusion/sparse-weight sweep (translations ON) | 2,637,645 | hybrid `--sparse-weight {0.2,0.3,0.5,0.7,1.0}`, `--fusion dbsf`, each `--translate-queries` | best w=0.7: overall nDCG +0.004 (gate ≥+0.02) | w=0.7 trades XL nDCG −0.038 for KA +0.015; w=1.0 XL nDCG −0.080 (G2 fail); dbsf 0.198 < RRF 0.211; low w collapses KA | — | **NO KEEP** (config unchanged: RRF, unweighted) | slice trade-off persists even with translations; per-translated-weight composite projects +0.017 < gate → second variant not spent. prefetch/ef/rescore latency knobs not run (serving bottleneck is the reranker → I7) |
| 2026-07-10 | **I4 agentic playbook + abstention + existence-check** | 2,637,645 | docstrings only; calibration `scripts/calibrate_min_score.py` rc=50, translations on, GPU | gate = G5 only: 264 tests pass, ruff clean, fingerprint unchanged, retrieval untouched | tool descriptions verified via `mcp.list_tools()` (all markers render); CLI smoke sane | — | **KEPT** | abstention threshold **0.92** from calibration (top-1 hit p1=0.926; scores saturate — misses median 0.997, so identity check carries the contract; only 10% of misses fall below 0.92). RERANK_MIN_SCORE stays 0.3 (calibration measured top-1 only). ⚠ MCP server must be reconnected (/mcp) to serve the new docstrings |
| 2026-07-10 | **I7 ONNX int8 reranker** | 2,637,645 | rerank@50, translations on, `RERANK_BACKEND=onnx` (CPU) vs fp32 GPU reference | **nDCG 0.320 vs 0.320 (Δ=0.000, gate ≤0.01) · rerank p50 28.2 s vs ~72 s fp32 CPU (0.39×, gate ≤0.5×)** | all slices within noise (XL nDCG −0.002, citation R@5 +0.059); probe: top-10 overlap 9–10/10, ρ≥0.97 | — | **KEPT** (knob, default torch) | 570 MB int8 via `scripts/export_onnx_reranker.py` (torch exporter; optimum refused — would downgrade transformers <5). int8 scores drift ≤0.1 → re-calibrate abstention before making onnx the serving default |
| 2026-07-11 | **I5 golden_set_v2 (partial)** | 2,638,482 | measurement tooling — not a retrieval change | n/a (grows the yardstick, doesn't move it): v2 = 337 verified pairs vs v1 103 (natural_question 75, legal_citation 67, cross_lingual 60, keyword 55, temporal 40, paraphrase 40) | v2 hybrid slice signal @2,638,482: temporal 0.795 nDCG (easiest), paraphrase 0.105 (hardest, by design zero content-word overlap), citation 0.393, XL 0.382; v1 consistency: eval_set_hash 985e1bc3e5cbf51e frozen, recall10 0.359 reproduces | — | **LANDED (partial); gating stays v1** | loader invariants green (reground/lint 337/337); v1 path byte-identical (no eval_set knob), v2 folds `eval_set:v2` → distinct config_hash; batches 1-5 + b3 first-pass committed; b6/b3-rework/b7/b8 pending in `.golden_v2_pending/` (subagent session limit). Commits 4eb2e0e→64fea5a |

---

*Evidence sources for the queue (verified 2026-07-09): BordIRLines arXiv:2410.01171 (BGE-M3 Georgian
native-only bias) · Poly-Vector Retrieval arXiv:2504.10508 (citation-query embedding failure) ·
Bruch et al. TOIS arXiv:2210.11934 + OpenSearch blog (weighted fusion > RRF) · BGE-M3 paper
arXiv:2402.03216 (w_sparse=0.3) · IRCoT arXiv:2212.10509, Self-Ask arXiv:2210.03350, Search-o1
arXiv:2501.05366 (agentic retrieval gains) · Google "Sufficient Context" arXiv:2411.06037
(prompted abstention ~93%) · Anthropic Contextual Retrieval (−35–49% failures) · LegalBench-RAG
arXiv:2408.10343 (rerankers are domain-contingent) · Webber/Moffat/Zobel CIKM'08 (≥150 topics) ·
RGB arXiv:2309.01431 (unprompted refusal fails). Anti-recommendations: multi-query arXiv:2603.02153,
HyDE-low-resource arXiv:2212.10496.*
