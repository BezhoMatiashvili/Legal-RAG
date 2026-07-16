# Georgia Legal Search — Simple Specs

This project searches Georgian laws and court documents. Its basic flow is:

`document -> chunks -> embeddings -> Qdrant -> hybrid search -> reranking -> MCP result`

The values below are the repository defaults, not machine-specific `.env` overrides. A
deployment can change them with environment variables.

## Chunking

We use **structure-aware, token-based chunking with overlap**.

- A document is split at Markdown headings, chapters, and legal article markers such as
  `მუხლი N`.
- Large sections are split by paragraph, then sentence, and finally by words when needed.
- Each chunk is at most **512 BGE-M3 tokens**.
- Neighboring chunks share up to **80 tokens** of overlap.
- The chunker tries to avoid a very small final chunk by using **64 tokens** as its minimum
  target.
- The clean original text, character offsets, legal structure, and a text hash are stored
  with every chunk.

Why: legal meaning often depends on the article, chapter, or nearby sentence. Structural
boundaries keep related text together, while overlap prevents important context from being
lost at a chunk boundary. The 512-token limit also keeps chunks inside the model's input
budget.

## Embedding

Before embedding, we add a small context header containing the document title, document
type, and heading/article path. This header helps the model understand a short clause. It is
used only for the embedding; the text shown to users stays clean.

Each chunk is embedded by `BAAI/bge-m3` in one pass as:

- a **1024-dimensional dense vector**, which matches meaning and paraphrases;
- a **learned sparse vector**, which matches important words, legal terms, numbers, and
  identifiers.

Both vectors are stored in Qdrant. Queries are embedded with the same model. Qdrant searches
the dense and sparse vectors and combines their rankings with Reciprocal Rank Fusion (RRF).

Why: dense search is good at meaning, while sparse search is good at exact terminology.
Using both is safer for legal search than relying on either one alone.

For an English query, the normal search route uses the dense branch only because English
sparse tokens do not match Georgian legal text well. The strict answer pipeline can also use
a protected Georgian translation branch when a private translator is configured.

## Models

| Model | Used for | Why |
|---|---|---|
| `BAAI/bge-m3` | Token counting, document embeddings, and query embeddings | It is multilingual and produces both dense and learned sparse vectors. This works well for Georgian semantic and exact-term search. |
| `BAAI/bge-reranker-v2-m3` | Reranking search candidates | It reads the query and each candidate chunk together, so it can order similar legal passages more precisely than vector search alone. |
| Deployment-provided private models | Translation and answer drafting for `legal_ask` | No answer-generating model is built in. A deployment must inject pinned private providers and a held-out risk calibrator; otherwise `legal_ask` returns a structured abstention instead of an unverified answer. |

The default retrieval process gets up to **80 candidates**, reranks them, and removes hits
below the default relevance score of **0.3**. This score measures passage relevance; it is
not legal confidence or proof that an answer is correct.

## MCP tools

The MCP server is called `legal_rag`. All of its tools are read-only and non-destructive.

| Tool | What it does | Why it exists |
|---|---|---|
| `legal_ask` | Answers, asks for clarification, or abstains after checking canonical evidence, document versions, quotations, and claims. | It is the safest tool for a final legal answer. |
| `legal_get_context` | Resolves an evidence ID to its exact text and nearby chunks. | It provides complete, verified context instead of a shortened search preview. |
| `legal_search` | Runs hybrid semantic search and reranking, with optional filters. | It finds relevant passages for research and debugging. |
| `legal_get_document` | Rebuilds a full document from its ordered chunks. | It lets the user read the complete source after search finds a useful passage. |
| `legal_lookup` | Finds a document by an exact document number, registration code, or document ID. | It avoids semantic guessing when the identifier is already known. |
| `legal_browse` | Lists documents using filters such as source, court, type, date, status, party, or keyword. | It supports corpus browsing when semantic search is unnecessary. |
| `legal_collection_info` | Shows collection size, vector settings, models, and available sources. | It explains what data can be searched and helps choose filters. |
| `ingest_status` | Shows index counts, source freshness, watcher progress, and the latest ingest report. | It checks whether the corpus is current and ingestion is healthy. |
| `legal_get_document_versions` | Lists a document's amendment/version history and marks the current in-force version when known. | It helps prevent using the wrong historical version of a law. |
| `legal_health` | Checks server, Qdrant, and immutable corpus-generation readiness. | It confirms that the search service is ready before it is trusted. |

## Serving modes

- **Local (default):** embedding, Qdrant search, and reranking all run locally.
- **Local search + remote GPU reranking:** embedding and Qdrant stay local, while the slow
  reranking step runs on a GPU worker.
- **Fully remote:** the local MCP server acts as a small client for a RunPod serverless
  worker that contains the models and Qdrant.

The normal path uses the same retrieval rules in every mode. If the optional remote
reranker fails, research search can clearly report a degraded fallback to RRF order.
