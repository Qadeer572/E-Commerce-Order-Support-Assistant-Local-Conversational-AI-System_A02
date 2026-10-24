# Assignment 2 RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Coding assistants may help development; the delivered chatbot must not use runtime agents or external tools.

**Goal:** Extend ShopBot with offline document retrieval, grounded answers, unchanged token streaming, measurable performance, and graceful failures.

**Architecture:** A local embedding model and persistent vector store supply evidence to the existing conversation manager. Retrieve before answering, apply the domain guard, and assemble one bounded prompt containing instructions, selected evidence, memory, and the current message. Business answers use retrieved documents rather than the current direct JSON/policy lookup path.

**Tech Stack:** Existing FastAPI, asyncio, Ollama/Qwen2.5, and HTML/CSS/JavaScript; add sentence-transformers, CPU PyTorch, transformers tokenizer support, and Chroma. Pin mutually compatible versions after testing a clean project environment.

**Spec:** [Assignment 2.md](../../../Assignment%202.md). This document is a proposed implementation plan, not a claim that RAG exists or benchmarks have been measured.

## Global Constraints

- Continue the same E-Commerce Order Support business use case and Assignment 1 codebase.
- Corpus requirement: **50–100 domain documents**; planned collection: **60 documents**. Documents and chunks are different counts.
- Use local CPU embeddings and a CPU vector store. All indexing and runtime operations work offline after initial dependency/model preparation.
- External tools and runtime agents are not allowed; business evidence must come through RAG.
- On every valid user query, embed/search the query or use a cached equivalent. Request **top-k = 3**, even if filtering, relevance, or context limits leave fewer usable chunks.
- Keep `/ws/chat` input `{session_id, message}` and output `token`, `done`, and `error` frames unchanged.
- Aim for retrieval below **one second**; report actual numbers independently from generation.
- Include caching, concurrent-user responsiveness, weak-match handling, embedding/store failures, timeouts, and context overflow handling.
- Choose **at most one bonus**. Visible source citations are the proposed optional bonus.
- Submit code, corpus, tests, and complete README through a **private GitHub repository** or ZIP. Optional Loom demo.

The assignment states a deadline of **October 8, 2026, 11:59 pm**, which precedes the current client date, October 10. No revised deadline is present in the supplied file. Grading weights are correctness/completeness 50%, viva 25%, and real-time behavior/performance 25%.

## Repository Review and Requirement Coverage

Baseline verification on October 10: **87 existing tests passed** using the available Python 3.13 virtual environment. The default Python 3.14 environment lacks pytest-asyncio; establish a project-local environment before adding dependencies.

| Requirement or current gap | Current state | Planned work |
| --- | --- | --- |
| Domain documents | 18 product records, 15 order records, one policy file; no RAG corpus | Task 1 |
| Incremental indexing | No embedder or vector store | Task 2 |
| Query retrieval and caching | Direct `data_lookup.py` calls | Task 3 |
| Context limits | Character/4 estimates; current message/output not fully budgeted | Task 4 |
| Conversation summary | Manager constructs it, but prompt builder does not receive it | Task 4 |
| Durable memory | Summarization replaces history, then disk save overwrites the transcript; summaries/entities are not restored | Task 4 |
| Session reset | Does not clear `conversation_summary` | Task 4 |
| Grounded streaming and failures | Streaming works; no RAG failure paths | Task 5 |
| Measurements and submission | README still says “no RAG,” contains latency placeholders, and describes old persistence behavior | Task 6 |
| Optional visible citations | No source display | Task 7 |

## Design Decisions

### Corpus and retrieval

Use UTF-8 Markdown only initially: **18 product specification documents + 18 distinct product FAQs + 15 fictional order summaries + 9 policy documents = 60**. Policy topics: returns, refunds, exchanges, shipping fees, delivery timelines, shipping destinations, damaged/wrong items, cancellation limits, and tracking guidance. Derive facts from existing fixtures; manually review FAQs and policies for consistency. Do not count a manifest, copied duplicates, or chunks toward the document target.

Use `sentence-transformers/all-MiniLM-L6-v2`. Its model card describes 384-dimensional embeddings and default truncation beyond 256 wordpieces, so start with **220 tokenizer tokens per embedding input, including title/heading prefixes, and 32-token overlap**. Preserve section boundaries where possible; split oversized sections deterministically. These are chosen defaults, not assignment-mandated values. [Model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

Use local `chromadb.PersistentClient`, explicitly supplied normalized embeddings, and cosine distance. Keep model files under ignored `models/` and the generated index under ignored `data/vector_store/`; commit the corpus and rebuild instructions. Chroma supports disk persistence, queries, and updates/upserts; use the configuration syntax of the version actually pinned. [Client reference](https://docs.trychroma.com/reference/python/client), [queries](https://docs.trychroma.com/docs/querying-collections/query-and-get), [updates](https://docs.trychroma.com/docs/collections/update-data).

Start with cosine similarity **0.45** (`similarity = 1 - cosine_distance`), then calibrate against a labeled development set. Do not report this initial threshold as proven. For order questions, apply an **exact canonical order-ID metadata filter**; an unknown order must never retrieve another customer's record. Other queries exclude order documents. Social messages and off-topic requests still attempt retrieval to satisfy the assignment wording, but ignore evidence and preserve the appropriate greeting/redirect behavior.

### Prompt budget and memory

Explicitly set Ollama `num_ctx=4096`; reserve `num_predict=512` plus **256 tokens of safety margin**, leaving at most **3328 prompt tokens**. Use the locally cached Qwen tokenizer and its chat template, checking the actual Ollama template before accepting the count. Tokenizer/model preparation is separate from runtime. [Qwen model and tokenizer example](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).

Reserve instructions and current user input first. Cap evidence at 1400 tokens, summary at 256, and recent history at 800, then enforce the total budget dynamically; these caps are not promises that everything fits. Keep the newest complete exchanges when possible, drop low-ranked evidence/old history first, and never exceed the hard limit to preserve a minimum message count. If instructions plus input alone cannot fit, return a bounded clarification without calling the model.

Keep the full transcript separate from the prompt's working history. A summary cursor tracks which older turns have already been summarized. Persist and restore the summary, cursor, intent, and last entities alongside messages, retaining compatibility with existing history files. Summarization runs only when necessary, with a bounded timeout and trimming fallback.

### Real-time operation

Load models/store once at startup; use local-only model loading and no runtime downloads. Perform blocking embedding/search in a dedicated executor with **2 workers**, at most **8 admitted jobs**, and a **1.5-second total retrieval deadline including queue time**. These limits are initial CPU settings to benchmark. A timed-out thread may continue: retain its admission slot until it actually finishes, rather than creating unbounded replacement work.

Cache results in a bounded **128-entry LRU with 300-second TTL**. Keys include effective query, entity/type filters, model revision, index revision, top-k, and threshold. Do not cache transient failures; do not share generated replies between sessions. Scope same-session turn processing with a per-session async lock, so separate browser tabs cannot interleave memory updates.

## Planned File Boundaries and Interfaces

| File | Responsibility |
| --- | --- |
| `data/corpus/**/*.md`, `data/corpus/manifest.json` | Source documents and metadata/provenance |
| `backend/retrieval/types.py` | Shared immutable document, chunk, hit, and result records |
| `backend/retrieval/corpus.py` | Loading, validation, deterministic token chunking |
| `backend/retrieval/embeddings.py` | CPU/local-only model loading and encoding |
| `backend/retrieval/store.py`, `indexer.py` | Chroma operations and incremental indexing |
| `backend/retrieval/service.py` | Query preparation, relevance, bounded async work, cache |
| `backend/conversation_manager/prompt_builder.py`, `context_manager.py` | Actual prompt assembly and memory budget |
| Existing `manager.py`, `session.py`, `main.py`, `history_store.py` | RAG integration, state, lifespan, persistence |
| `scripts/prepare_models.py`, `validate_corpus.py`, `index_documents.py` | Repeatable local preparation/indexing commands |
| `scripts/benchmark_rag.py`, `data/evaluation/*.json`, `docs/benchmarks/` | Quality labels, measurements, reproducible results |

Shared records in `types.py`:

- `Document(doc_id: str, title: str, source: str, text: str, doc_type: str, sku: str, order_id: str)`; empty strings represent absent entities. Source paths are relative.
- `Chunk(chunk_id: str, document: Document, text: str, ordinal: int)`.
- `Hit(chunk_id: str, source: str, title: str, text: str, similarity: float, doc_type: str, sku: str, order_id: str)`.
- `RetrievalResult(status: Literal['ok','no_match','unavailable','timeout','busy'], hits: list[Hit], elapsed_ms: float, cache_hit: bool)`.
- `IndexReport(added: int, updated: int, deleted: int, unchanged: int, revision: str)`; counts refer to documents.

## Review Focus

1. Unknown order IDs and identical follow-ups from different sessions: return no match and prevent cross-order evidence/cache reuse (Tasks 3, 5).
2. Changed, shortened, or deleted source documents: remove obsolete chunks and invalidate cached evidence (Task 2).
3. Long Unicode input, oversized evidence, and lengthy memory: enforce the complete prompt budget (Task 4).
4. Missing model/index or permanently slow work: bound user waiting and protect other sessions/health requests (Tasks 3, 5).
5. Retrieved text containing instructions or contradictory facts: treat it as evidence, preserve system rules, and acknowledge uncertainty (Tasks 1, 5).

## Task 1: Build a Validated Corpus and Chunker

**Files:** Create `data/corpus/{products,faqs,orders,policies}/`, `manifest.json`, `backend/retrieval/{__init__,types,corpus}.py`, `scripts/{prepare_models,validate_corpus}.py`, `tests/test_corpus.py`. Modify `requirements.txt`, `.env.example`, `backend/config.py`, `.gitignore`.

**Interfaces:** Produce `load_documents(root: Path) -> list[Document]` and `chunk_document(document: Document, tokenizer, *, max_tokens: int = 220, overlap: int = 32) -> list[Chunk]`. Model preparation caches pinned embedding and Qwen tokenizer revisions; save revisions in a local preparation manifest.

- [ ] Write failing tests: `test_corpus_has_60_unique_documents` asserts `len(documents) == 60` and unique IDs; `test_chunks_preserve_order_id_and_fit_embedding_limit` asserts every embedding input is at most 220 tokens and the expected order ID remains intact.
- [ ] Run `python -m pytest tests/test_corpus.py -v`; confirm missing-module/behavior failures before implementation.
- [ ] Establish a project-local virtual environment, test compatible dependency versions, and pin them. Prepare local models once; verify later loading works with network access disabled. Author/review all 60 documents and manifest entries, preserving fixture prices, stock, order status, and policy facts.
- [ ] Implement loading/chunking and reject empty documents, duplicate IDs, contradictory structured fixture facts, invalid paths, and missing metadata. Stable chunk IDs use document ID, content hash, ordinal, and chunker version.
- [ ] Run the tests, `python scripts/validate_corpus.py`, and `python scripts/seed_data_check.py`; expect 60 valid source documents and no fixture errors. Commit only this task's files with `Add validated RAG corpus and token chunking`.

## Task 2: Implement Incremental Offline Indexing

**Files:** Create `backend/retrieval/{embeddings,store,indexer}.py`, `scripts/index_documents.py`, `tests/test_indexer.py`.

**Interfaces:** Consume Task 1 records/chunks. Produce `LocalEmbedder.encode(texts: list[str]) -> list[list[float]]`; `VectorStore.upsert(chunks: list[Chunk], embeddings: list[list[float]]) -> None`; `delete_document(doc_id: str) -> None`; `search(embedding: list[float], *, top_k: int, filters: dict) -> list[Hit]`; and `index_corpus(corpus_dir: Path, store: VectorStore, embedder: LocalEmbedder) -> IndexReport`.

- [ ] Write failing tests: an unchanged second index run asserts `report.unchanged == 60` and unchanged chunk count; a shortened/removed document asserts obsolete chunk IDs are absent. An injected embedding failure must leave the last complete index revision usable.
- [ ] Run `python -m pytest tests/test_indexer.py -v`; confirm expected failures.
- [ ] Implement normalized CPU embeddings and disk persistence with explicit embeddings, cosine configuration, and no default embedding downloads. Track file hashes, model revision, dimension, and chunking settings in the index manifest.
- [ ] Compute all changed-document embeddings before the first store mutation, so embedding failures preserve the previous complete index. Then upsert changed documents, remove obsolete/deleted chunks, and publish the revision manifest only after success. A persistent incomplete-build marker makes interrupted store updates unavailable until recovered. Re-indexing and a running server must not write/read the same index concurrently; reject an index update while the server owns it.
- [ ] Run `python scripts/index_documents.py` twice, restart the index reader, and run the tests; the second run performs no duplicate inserts. A model/chunker change requires an explicit rebuild rather than mixing incompatible embeddings. Commit with `Add incremental local vector indexing`.

## Task 3: Build Retrieval, Relevance Filtering, and Cache

**Files:** Create `backend/retrieval/service.py`, `data/evaluation/retrieval_dev.json`, `tests/test_retrieval.py`.

**Interfaces:** Consume embedder/store and index revision. Produce `Retriever.retrieve(query: str, *, doc_types: tuple[str, ...], sku: str | None = None, order_id: str | None = None, top_k: int = 3) -> RetrievalResult` as an async method; `Retriever.close() -> None`; and `build_retrieval_query(user_message: str, session: Session, intent: str) -> str`.

- [ ] Write failing tests: unknown `ORD-99999` gives `status == 'no_match'` and `hits == []`; two queries with different order filters never share evidence; weak scores return no match; a changed index revision invalidates a repeated-query cache result; timeout and overload complete within the configured deadline.
- [ ] Run `python -m pytest tests/test_retrieval.py -v`; confirm failures.
- [ ] Implement top-3 search, exact order/type filters, normalized similarity, threshold filtering, deduplication, and contextual expansion only for recognized follow-ups. Use stored/retrieved entity metadata, not a new direct business-data lookup. Distinguish genuine follow-ups from a new unrelated topic.
- [ ] Add bounded executor admission, deadline, and cache behavior defined above. Load dependencies once; represent missing/incompatible index/model and search exceptions with explicit statuses. Retain slots for unfinished timed-out jobs and cancel queued jobs during shutdown.
- [ ] Build a development set with 20 answerable and 10 unanswerable questions, expected source IDs, and exact-entity cases. Calibrate the threshold without using the later holdout set; run unit tests plus a real local-model/store smoke test. Commit with `Add bounded retrieval and query caching`.

## Task 4: Enforce Prompt Limits and Repair Working Memory

**Files:** Modify `backend/conversation_manager/{prompt_builder,context_manager,context_trimmer,session}.py`, `backend/{config,llm_engine,history_store,main}.py`. Create `tests/{test_prompt_budget,test_memory}.py`.

**Interfaces:** Produce `build_messages(history: list[Turn], user_message: str, intent: str, *, evidence: list[Hit], summary: str | None, tokenizer, context_tokens: int = 4096, response_tokens: int = 512) -> list[dict]`. Budget overflow of mandatory input raises `PromptBudgetError`. Session adds `summarized_turn_count: int`; persisted state includes memory/entity fields while preserving existing REST response shape.

- [ ] Write failing tests: complete chat-template count plus 512 output and 256 safety tokens is `<= 4096`; oversized Unicode/current input raises the bounded error; an old summary fact appears in the assembled prompt; summarization leaves the original transcript intact; reset clears summary/cursor/entities; old and new history formats load correctly.
- [ ] Run `python -m pytest tests/test_prompt_budget.py tests/test_memory.py -v`; confirm failures.
- [ ] Implement the budget strategy above, including evidence title/source labels and current input. Set Ollama `num_ctx=4096` and `num_gpu=0` explicitly. Verify the tokenizer/template against the locally served model and reserve overhead; do not use embedding-token counts as generation-token counts.
- [ ] Refactor the manager's unused prompt assembly into the single real builder. Use a temporary working-history view and summary cursor, preserve the full persisted transcript, and restore working state on resume. Bound summary generation and fall back to trimming if it fails.
- [ ] Run the complete suite and a long-chat/resume/reset smoke test; existing scope behavior must remain intact. Commit with `Enforce RAG prompt budgets and preserve conversation memory`.

## Task 5: Integrate RAG with Streaming and Graceful Failures

**Files:** Modify `backend/main.py`, `backend/conversation_manager/{manager,intent_triage,domain_policy}.py`, `.env.example`, `backend/config.py`. Create `tests/{test_rag_conversation,test_rag_websocket}.py`; adapt existing conversation tests to inject a retriever double.

**Interfaces:** `ConversationManager(llm_engine, retriever, tokenizer)` consumes Tasks 3–4. Existing `handle_turn(session: Session, user_message: str) -> AsyncIterator[str]` and WebSocket frames remain unchanged. Lifespan constructs one retriever and closes it on shutdown.

- [ ] Write failing tests: retrieved facts unsupported by model training appear in a reply; `hi` remains conversational; the Pakistan-capital question redirects; unknown orders never expose another order; retrieval failure sends a safe answer followed by `done`; streams contain multiple `token` frames before `done`. Retrieved “ignore previous instructions” text must not override domain rules.
- [ ] Run `python -m pytest tests/test_rag_conversation.py tests/test_rag_websocket.py -v`; confirm failures.
- [ ] Retrieve/cache on each accepted message, then apply social/domain handling. Extend recognized support vocabulary for corpus questions such as warranty coverage; retain unrelated-topic guards and safe follow-up routing. Remove direct product/order/policy facts and no-match fallback injection from the manager; retain fixture helpers only for corpus validation/metadata needs.
- [ ] Delimit retrieved evidence as untrusted document text. Instruct the model to use only supplied evidence, identify missing information, and never invent order actions or facts. Contradictory evidence produces an uncertainty response. No match says “I couldn't find reliable information for that in our store documents” and offers a relevant clarification; unavailable/timeout/busy states explain temporary unavailability. Greetings remain natural and unrelated questions stay redirected.
- [ ] Handle `PromptBudgetError` with a short clarification; record backend-only timings/statuses. Add retrieval readiness to the health response without changing its existing keys. Offload synchronous history-file operations and protect same-session turns with a lock; separate sessions retain independent state.
- [ ] Run the complete suite. Inject missing index, embedding/store exceptions, timeout, and slow retrieval while checking another session and `/api/health` remain responsive. Verify offline operation with prepared models, and manually exercise the light UI, streaming, reset, and history resume. Commit with `Ground streaming support answers in retrieved documents`.

## Task 6: Measure Quality and Performance; Complete Submission Docs

**Files:** Create `scripts/benchmark_rag.py`, `data/evaluation/retrieval_holdout.json`, `docs/benchmarks/rag-results.json`, `docs/ai-prompts.md`. Modify `README.md`, `AGENTS.md`.

**Interfaces:** Benchmark CLI consumes Task 3 retrieval plus the existing WebSocket endpoint. Report records include hardware, versions, corpus/chunk count, model revisions, threshold/top-k, embedding/search/queue/end-to-end retrieval milliseconds, cache status, first-token latency, and generation/total time.

- [ ] Write failing benchmark tests in `tests/test_benchmark_rag.py`: fixture durations produce the expected p50/p95 summaries, and cached results are separated from uncached retrieval. Async concurrency tests assert that five sessions preserve IDs and each receive terminal frames without event-loop starvation.
- [ ] Run the benchmark tests; confirm failures before implementing aggregation.
- [ ] Prepare an unseen holdout set of 20 answerable and 10 unanswerable queries; evaluate source hit-rate at top-3, exact-order correctness, false-positive/no-match rates, and manually checked answer grounding. Do not substitute mocked embeddings or cached query timings for real retrieval measurements.
- [ ] Run `python scripts/benchmark_rag.py --queries data/evaluation/retrieval_holdout.json --iterations 100 --concurrency 1`, then repeat with `--concurrency 5`. Include warm uncached, repeated cached, and cold startup results separately; record embedding/search separately from generation. Aim for warm retrieval p95 below 1000 ms; report misses honestly and optimize only measured bottlenecks.
- [ ] Rewrite README for Assignment 2: model/store rationale, exact chunking/top-k/threshold, corpus provenance/count, architecture diagram, offline setup/index/update commands, real benchmark table, failure examples, context/memory strategy, tests, known limitations, and team attribution. Remove “no RAG,” fake action claims, stale persistence text, and benchmark placeholders. Record actual AI prompts used during implementation in `docs/ai-prompts.md` for viva reproduction.
- [ ] Run `python -m pytest`, both data validators, an index-update smoke test, and the documented clean setup. Confirm the submission includes corpus, pipeline, retrieval integration, tests, and complete README; excludes `.env`, real chats, and model weights. Commit with `Document and benchmark Assignment 2 RAG delivery`.

## Task 7: Optional Bonus — Visible Sources

Only start after Tasks 1–6 meet their acceptance checks. Do not also implement hybrid search or re-ranking for bonus credit.

**Files:** Modify `backend/conversation_manager/manager.py`, `frontend/style.css` only if needed; create `tests/test_source_citations.py`; update README/demo examples.

**Interfaces:** Append a readable `Sources: <relative document titles/paths>` footer through ordinary `token` frames before `done`; use only the evidence actually included in the prompt. No new frame type, endpoint, or citation fetch is needed.

- [ ] Write failing tests: discarded/weak chunks never appear as sources, repeated source paths are deduplicated, and greeting/no-match/redirect replies have no source footer.
- [ ] Run `python -m pytest tests/test_source_citations.py -v`; confirm failures.
- [ ] Append server-controlled sources for grounded answers, distinct from any model-generated references. Escape display text through the existing Markdown renderer; show no local absolute paths.
- [ ] Run the full suite and visually check the source footer in the light UI on desktop/mobile. Document this as the single chosen bonus. Commit with `Show retrieved document sources in chat`.

## Execution Order and Completion Gate

Run Tasks **1 → 2 → 3 → 4 → 5 → 6**, then optionally 7. This is one cohesive extension; the corpus/index and online retrieval/prompt integration are sequential milestones, not independent new applications. Preserve existing uncommitted user work; inspect `git status` and stage task-specific files only.

Completion means a clean offline rebuild from committed source documents, grounded streamed answers, reliable no-match/failure handling, verified concurrency and context limits, measured retrieval performance, and every required README item. It does not mean only that the demo answers one question. Proposed execution: native implementation in this session, with review at integration and final delivery.
