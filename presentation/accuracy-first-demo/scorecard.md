# Accuracy-first presentation scorecard

> This is an accuracy-first pre-production Georgian legal RAG. It answers only when evidence, identity, version, quotation, and completeness checks pass; otherwise it clarifies or abstains.

## Measured presentation results

| Measure | Result |
|---|---:|
| Total frozen questions | 20 |
| Expected-answer exact-identity questions | 15 |
| Correct document identity | 15 / 15 |
| Required identity evidence found | 15 / 15 |
| Official URL present and matches frozen source | 15 / 15 |
| Ambiguities correctly clarified | 2 / 2 |
| Incomplete sources correctly abstained | 2 / 2 |
| Broad advice correctly abstained | 1 / 1 |
| Deterministic result repeats | 20 / 20 |

Canonical quotation validation: `not_available_in_legacy_corpus`. The separate frozen snapshot audit
reconstructed 15
of 15 expected-answer spans and matched
15 frozen
document hashes. That audit is not canonical-generation evidence.

Degraded/failed query IDs: run 1 `[]`; run 2 `[]`.

Latency: run 1 median 14.398 ms / p95 52.487 ms;
run 2 median 7.99 ms / p95 19.832 ms.
Read-only retrieval only: run 1 median 17.103 ms / p95
60.262 ms; run 2 median
8.296 ms / p95 24.581 ms.

## Identity and hashes

- Question set SHA-256: `61838610b10f4c319f22516405e0a1828b4bade019e97b356726d02e49fd64dd`
- Configuration SHA-256: `cacaec91eedef93ad35df4f6dec72eea4cb2fe7235e412e66d1081f6808114b3`
- Corpus observation SHA-256: `2046bdca5dd2ab79eefd656e7b8955dc695a084fa0e745cb1fce325f0796e3b3`
- Run 1 decision SHA-256: `288d4eaede56296331668140852df04cba6b395eafe6a96b38d3ce04ba7102e8`
- Run 2 decision SHA-256: `288d4eaede56296331668140852df04cba6b395eafe6a96b38d3ce04ba7102e8`
- Immutable generation identity: `not_available_in_legacy_corpus`

## Fresh test evidence

- `presentation_tests`: passed (37 passed, 0 failed, 0 deselected) — Hermetic builder, safety, atomicity, hashing, privacy, and offline rehearsal tests.
- `legal_answer_evidence_abstention`: passed (271 passed, 0 failed, 0 deselected) — Focused legal-answer, evidence, retrieval, release-review, and abstention regressions.
- `root_tests`: passed (281 passed, 0 failed, 0 deselected)
- `ingest_non_snapshot_tests`: interrupted — Aggregate stopped after 42%; isolated test_part3_mcp versions test stalled on its second asyncio.to_thread call.
- `ruff_root`: passed — All checks passed.
- `ruff_ingest`: passed — All checks passed.
- `symbol_map`: passed — Generated and checked: memory refs OK.
- `git_diff_check`: passed
- `git_diff_cached_check`: passed

## Interpretation boundary

Retrieval identity, frozen-span integrity, synthetic contract validation, and legal
correctness are separate measures. They are not collapsed into one accuracy figure.

Legal correctness review: not yet independently adjudicated.

The presentation set is small and not a blind benchmark. Production severe-error
certification is not yet complete. The full immutable corpus and lawyer-reviewed v3 release
study remain future work.
