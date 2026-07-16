# Reviewer checklist

- [ ] `questions.sha256` matches `questions.jsonl` before discussing results.
- [ ] Both runs contain all 20 IDs in the frozen order.
- [ ] Per-question result hashes and run decision hashes repeat.
- [ ] Exact successes match the annotated source/document identity and include an official URL.
- [ ] q169, q291, and q329 frozen quotes reconstruct from their saved offsets and hashes.
- [ ] q024 contains no quotation, body text, promoted metadata, or party data.
- [ ] q121 emits clarification and q024 emits abstention; neither emits answer text.
- [ ] Canonical legacy fields say `not_available_in_legacy_corpus`.
- [ ] Synthetic contract fixtures are labelled synthetic and excluded from measured retrieval.
- [ ] q191, q229, and q276 visibly retain their repealed/historical status.
- [ ] Publication metadata dates are not described as act-adoption dates.
- [ ] Test evidence comes from fresh commands or is visibly `not_run`.
- [ ] No private query log, secret, or unnecessary party name is present.

Legal correctness review: not yet independently adjudicated.
