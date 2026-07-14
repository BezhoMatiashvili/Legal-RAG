---
name: georgia-legal-research
description: Research Georgian legislation and court practice through the legal_rag MCP tools. Use for legal questions, document or case lookup, consolidation history, source-backed summaries, and corpus coverage questions.
---

# Georgia Legal Research

Use the `legal_rag` MCP server as the source of record for corpus-backed answers. The corpus contains Georgian legislation and court practice; it is a retrieval system, not a legal authority or a substitute for professional advice.

## Research workflow

1. For a new thread or suspected outage, call `legal_health` or `legal_collection_info` before a costly search.
2. Use `legal_lookup` for exact document numbers or registration codes. Use `legal_search` for questions, concepts, paraphrases, and cross-lingual queries. Use `legal_browse` for filtered exploration.
3. Start with a narrow query and relevant source, document-type, language, status, or date filters. Broaden only when results are sparse.
4. Open promising records with `legal_get_document`. When consolidation or historical wording matters, call `legal_get_document_versions` and state which version/date supports the answer.
5. Synthesize only claims supported by retrieved material. Cite the document title, identifying number when present, date/version when relevant, and `source_url`.

## Answer rules

- Distinguish the retrieved text from your interpretation.
- Never invent article numbers, holdings, dates, status, parties, or URLs.
- Treat low scores, conflicting versions, missing full text, or incomplete coverage as uncertainty and say so.
- Prefer primary documents over summaries. If sources conflict, explain the conflict and identify each source.
- Preserve Georgian legal terminology where precision matters; add a translation or explanation rather than silently replacing it.
- Do not imply that corpus coverage is exhaustive. Use `legal_collection_info` when coverage affects the conclusion.
- Include a brief not-legal-advice qualification when the user may rely on the answer for a consequential decision.

## Failure handling

If tools are unavailable, do not answer as though a search occurred. Report the failed prerequisite and direct maintainers to run `plugins/georgia-legal-search/scripts/health-check.sh` from the repository root. If retrieval returns no reliable evidence, state that outcome and suggest a narrower identifier, alternate Georgian phrasing, or filter change.
