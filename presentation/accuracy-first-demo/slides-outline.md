# Six-slide outline

## 1. Legal retrieval errors are high-impact

- Wrong identity, version, or quotation can change the practical meaning.
- The safe response to unsupported evidence is clarification or abstention.

## 2. Accuracy-first architecture

- This is an accuracy-first pre-production Georgian legal RAG. It answers only when evidence, identity, version, quotation, and completeness checks pass; otherwise it clarifies or abstains.
- Exact selector → identity gate → canonical evidence gate → answer gate.
- Today's live track stops at legacy identity retrieval; the tracks remain visibly separate.

## 3. Canonical evidence and version traceability

- Frozen offsets and hashes make the offline examples reproducible.
- Legacy canonical quotation/evidence IDs are `not_available_in_legacy_corpus`.
- A sealed immutable generation remains required for canonical production evidence.

## 4. Demonstration

- Exact identity: q169, q291, q329.
- Ambiguity clarification: q121.
- Incomplete-source abstention: q024.
- Static fallback uses persisted results and requires no network or model.

## 5. Measured scorecard and test reliability

- Identity: 15 / 15.
- Clarification/abstention reported by category, separately from identity and frozen-span checks.
- Repeat match: 20 / 20.
- Small curated presentation set; not a blind benchmark.

## 6. Honest status and roadmap

- Production severe-error certification is not yet complete.
- Full immutable corpus build and lawyer-reviewed v3 release study are future work.
- English live demonstration waits for production model attestation.
- Next milestone: blind legal adjudication against a sealed generation.
