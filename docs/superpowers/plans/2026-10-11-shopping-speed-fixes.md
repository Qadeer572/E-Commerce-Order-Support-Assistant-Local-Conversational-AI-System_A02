# Shopping and response latency fixes

Goal: Accept ordinary shopping requests, ground catalog browsing in indexed documents, and reduce local CPU generation delays.

Work in the existing checkout because it contains the uncommitted Assignment 2 implementation under review. Preserve all existing work.

- [x] Add failing routing tests for budget shopping, catalog browsing, product typos, and unrelated requests.
- [x] Normalize common product typos, broaden shopping intent, and route broad browsing to three indexed catalog summaries (top-k remains 3). Test summaries against fictional fixtures.
- [x] Add failing no-match tests; stream a grounded-safe conversational clarification for missing product/policy evidence, preserving exact unknown-order and unavailable-store responses.
- [x] Test Ollama request options; keep local CPU model resident, cap ordinary answers, and shorten system prompts. Compare real generation timings before and after.
- [x] Preserve original chunk text using character offsets; rebuild the index and recheck quality after discovering lossy MiniLM decoding.
- [x] Validate corpus, run all tests, verify live WebSocket streaming/citations and HTTP history/reset/deletion, and document evidence. Repair Windows history atomic-replace race.
- [ ] Rendered desktop/mobile checks: blocked because no browser surface exists and creating an in-app browser returned unavailable. HTTP/static checks passed; do not claim visual verification.
- [x] Exclude generated caches and temporary verification files from submission; provide a source ZIP. Do not claim submission or revised deadline.
