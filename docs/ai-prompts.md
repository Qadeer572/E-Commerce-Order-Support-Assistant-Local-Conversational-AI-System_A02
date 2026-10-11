# AI assistance provenance

User requests used during this session included:
- “Generate a file named AGENTS.md that serves as a contributor guide for this repository.”
- “ok tell me overview of the project like how memory managed”
- “backend is not runnung please run th backend”
- “same issue ... send button disable and still show connecting”
- Requests to allow natural greetings while redirecting unrelated questions, followed by “ok proceed to implemenation”.
- “our UI is dark and we have to make this light”, followed by approval.
- “review the assignment 2 in this folder and build plan”, followed by “please good to proceed”.

The implementation followed `docs/superpowers/plans/2026-10-10-assignment-2-rag.md`. Codex drafted corpus documents from fictional fixtures, wrote the local RAG pipeline and tests, repaired memory budgeting/persistence, and measured CPU retrieval. A separate AI reviewer was asked to inspect exact order/cache isolation, incremental updates, prompt limits, slow/missing assets, and untrusted evidence. Development assistance is not a runtime chatbot agent.

Reproduce with the README commands. Explain in the viva: cosine distance versus similarity; why MiniLM chunks fit its tokenizer; how exact order filters prevent cross-order retrieval; cache keys and admission limits; why retrieval latency differs from model generation; how the full transcript differs from the bounded prompt; and why prompt grounding is not a guarantee of model correctness.

## October 11 follow-up

The user requested fixing remaining Assignment 2 gaps, reducing response latency, and accepting shopping questions such as ?i need cheap product? and ?list if prodcut?. Codex added routing and price-filtering regressions, catalog summaries, original-text chunking, guarded no-match conversation, model warmup and CPU model comparisons, and repaired the Windows history-file race found during live HTTP checks. Browser automation was attempted but no browser surface was available.

Repeated live acceptance checks showed the 0.5B model could refuse catalog lists. Final catalog replies therefore stream exact selected rows from retrieved, prompt-budgeted evidence; no business fixture lookup or external tool is used. Full catalog and cheapest-item live assertions passed.
