# Accuracy-first answering contract

`legal_search` remains a retrieval/debugging tool. Its Markdown previews and reranker
scores are not an answering contract. The accuracy profile is exposed through:

```text
legal_ask(question, language?, as_of?, filters?, profile="strict")
legal_get_context(evidence_id, neighbor_chunks=1, max_tokens=12000)
```

`legal_ask` always returns JSON with `outcome=answer|clarify|abstain`. An answer contains
atomic material claims, explicit `quoted_law` versus `system_interpretation` kinds,
content-bound evidence IDs, exact quotations and source offsets/hashes, official identity
and effective-version metadata, the complete model/prompt/generation tuple, and a trace of
every retrieval branch and validation decision.

## Fail-closed path

The strict path is:

```text
intent/language/entity plan
→ original + protected Georgian translation candidates (depth ≥80)
→ exact identity + document/article-scoped + unrestricted branches
→ one global rerank (raw score is relevance only)
→ bounded canonical evidence pack (8–16K tokens)
→ atomic-claim composition
→ deterministic evidence/quote/hash/identity/version/completeness validation
→ at most one repair
→ held-out selective-risk calibration
→ answer, clarification, or abstention
```

The service refuses to answer when the language is unsupported, a private translator or
reranker degrades, exact identity remains ambiguous, an `as_of` version is missing or
uncertain, current-law freshness is unverified, evidence is incomplete/non-official, any
quotation/hash/offset fails, a material claim is unsupported, repair still fails, or the
selective-risk calibrator rejects the case. No score such as `0.92` is interpreted as an
answer probability.

The repository does not silently configure an external generator or translator. A private
deployment must install pinned providers and a held-out calibrator with
`mcp_server._install_answer_runtime(...)`. Until then `legal_ask` returns a structured
abstention without loading retrieval models. Provider version strings are part of the
answer trace and must identify the same model/revision covered by the production license
attestation; a display label is not a substitute for that binding.

## Corpus contract

Evidence-producing generations are immutable and must carry, per passage:

- generation, source fingerprint, normalizer/chunker/model revisions;
- official source authority/URLs, content completeness and extraction status;
- document/article/clause/chapter/heading/parent identity;
- exact character/page offsets, document content hash and passage hash;
- version/supersession/effective interval/consolidation/legal status metadata.

An evidence ID encodes and checksum-binds the active generation, Qdrant point ID, and exact
passage hash. `legal_get_context` re-resolves all three and returns full canonical text;
it never reconstructs evidence from the 700-character Markdown preview.

Current-law eligibility comes from a separately verified, immutable freshness report; it
is never inferred from a mutable boolean on a search hit. The report is bound to the
generation manifest, source observations, audit instant, source SLA and each
`(source, document_id, version_id)` record. `legal_ask` fails closed if the installed
freshness guard cannot verify a passing, unexpired report for the active generation. The
weekly audit entry point is `scripts/audit_corpus_freshness.py`; the supplied
`systemd/legal-corpus-freshness-audit.{service,timer}` writes owner-only reports and does
not mutate the collection.

## Release status and gates

The code contract is deliberately stricter than the current legacy collection. A new clean
generation, private models, calibration data, and blind v3 review are release inputs, not
values that code may guess. Production promotion remains blocked until the release-gate
evaluator proves all of these on a predeclared blind test:

- at least 1,000 answered outcomes;
- one-sided 95% severe-error upper bound below 1% and material-error upper bound below 3%;
- at least 50% coverage of in-scope answerable questions;
- 100% mechanically valid evidence IDs, quotations, offsets, and hashes;
- zero answers from degraded/stale/ambiguous/failed-validation paths;
- deterministic rankings and answers across two identical runs;
- no source/high-risk slice regression over two percentage points;
- private-profile p95 latency at or below 30 seconds.

Model bakeoffs require immutable revisions and explicit commercial/private-deployment/data
license attestations. Selection is lexicographic: severe error, material error, citation
support, temporal correctness, completeness, correct refusal, then latency and cost. In
production, each attestation is matched to the provider role, model ID, immutable revision,
and public version string; truthy strings are not accepted as license booleans. The strict
calibration artifact is immutable, checksum-verified from its source bytes, bound to the
complete evaluated pipeline/generation/configuration, and cannot relax the fixed 95%
confidence, 1% severe-error, or 3% material-error limits.

Candidate Recall@50/@80 is measured on the first 50/80 rows of one stable, deduplicated
pre-rerank union—not on 50/80 candidates from every branch independently. Reranker
fine-tuning is blocked unless positives are annotated evidence chunks, blind
document/version lineages are absent, and the mined set covers wrong passages in the
correct document, neighboring articles, similar authorities, and current/repealed
versions.

The strict release aggregate is derived by `eval.answer_release_review.aggregate_release_candidate`;
hand-written summary counts are not accepted. Its input is the sealed v3 blind dataset,
at least two complete candidate runs, two blinded independent reviews per output (plus a
third adjudication on disagreement), and a canonical evidence ledger exported from the
active generation. The resulting bundle cryptographically binds the dataset and blind-ID
set, every run's rankings and answers, reviews, deterministic mechanical audit, and the
exact canonical evidence subset. The strict gate,
`eval.release_gate.evaluate_release_candidate`, accepts the complete recomputable
`ReleaseEvidenceBundle`, not a caller-supplied aggregate. Error gates use the
more conservative of query-level and document/version-family cluster-level one-sided
bounds, so 1,000 correlated answers from a few authorities cannot manufacture precision.

Deployment transitions do not accept gate booleans. Shadow, 5% canary, and production
each require a checksum-bound release authorization for the exact seven-component serving
tuple and the same release evidence/policy chain. Any SLO breach atomically restores the
previously authorized tuple. Canary feedback is local-only and carries that same tuple,
authorization, traffic stage, supported language, and answer trace ID.

This repository currently implements and tests those contracts; it does **not** assert
that a release candidate has passed them. Promotion still requires a clean immutable v3
publish, a predeclared real-lawyer blind set yielding at least 1,000 answered outcomes,
independent adjudication, pinned private model bakeoffs, a held-out checksum-bound calibration
artifact, two deterministic production-path replays, and a passing release bundle. Until
those artifacts exist, production mutation remains blocked.
