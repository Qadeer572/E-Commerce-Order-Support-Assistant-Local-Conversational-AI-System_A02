**Assignment 2** 

**Due Date: Oct 8, 2026 11:59 pm** 

• This assignment may be completed **individually or in a group of exactly two students**. 

• No late submissions will be accepted. Submissions that do not comply with the specifications in this document will  not be marked and will receive a zero grade. 

• You may use generative AI tools to help with this assignment, but you should generate the minimal amount of code  needed to implement the required functionality. You must be able to **explain your code** and reproduce the prompts you used if asked to during the viva. **This is not a one-shot assignment: it requires several components to be built and tested separately before they come together into a working system. Use the tools, but make sure you  understand what they produced.** 

• Submit a single **private** GitHub repository link on Google Classroom containing all code, documentation, and test  cases, **or** a .zip file with the same contents. 

• A short Loom video demoing your system is optional but recommended. Your README.md must contain every  item listed in the Deliverables section of this document. A group that builds a working system but ships an  incomplete README will lose marks on completeness. 

• This assignment builds directly on your own group's codebase from the previous assignment in this series. You must  continue using the **same business use case** you selected earlier. 

**Introduction** 

In Assignment 1 you built a conversational assistant that relied purely on prompt orchestration and conversation memory.  That works for general dialogue, but a real business assistant also needs to answer questions grounded in specific  documents it was never trained on: policies, price lists, FAQs, catalogues. In this assignment you will give your assistant that ability using Retrieval-Augmented Generation (RAG), so it can look things up before it answers instead of guessing. 

**Constraint:** External tools and agents are still not allowed. You must perform everything in this assignment using RAG  only. 

**Objective** 

● Build an offline pipeline that chunks, embeds, and indexes 50–100 domain documents.   
● Retrieve the most relevant chunks for each user query and ground the response in them.   
● Keep the system's real-time streaming behaviour intact — retrieval should not make the assistant feel slower to  the user. 

● Fail gracefully when retrieval finds nothing useful, instead of forcing a bad answer. 

**System Architecture Overview** 

You are extending the Assignment 1 architecture with one new module. The WebSocket API contract from Assignment 1  must not change. 

Web UI \<--WebSocket--\> FastAPI Backend \<--\> Conversation Manager \<--\> Local LLM Engine 

| 

\+--\> Retrieval Module (Embedding+Vector 

Store)  
**Your Document Collection** 

Build a corpus of 50–100 documents relevant to the business use case you picked in Assignment 1 (PDF, plain text,  markdown, or HTML are all fine). As a starting point for your domain: 

• **Gym / Fitness Membership Assistant** — membership plans, class schedules, trainer bios, and gym policies • **Real Estate Rental & Leasing Assistant** — property listings, lease terms, and building policies • **Event & Venue Booking Assistant** — venue packages, catering menus, and booking terms • **Library Assistant** — the catalogue, borrowing policies, and fine rules 

• **Car Rental Assistant** — the fleet listing, pricing, and insurance terms 

• **E-Commerce Order Support Assistant** — product FAQs, return policy, and shipping information 

You are free to write these documents yourself, adapt real public examples, or generate a first draft with an AI tool and edit  it but, they must be realistic enough that grounded answers are actually meaningfully different from ungrounded ones. 

**Tasks** 

**Phase I — Indexing Pipeline** 

● Chunk your documents according to your chunking strategy.   
● Embed each chunk with a local, CPU-friendly embedding model (for example all-MiniLM-L6-v2 via  sentence-transformers, or a GGUF-compatible embedding model). 

● Store the embeddings in a CPU vector store: Chroma, FAISS, etc are acceptable. 

● Make the pipeline re-runnable, so you can update your document set later without starting from scratch. **Phase II — Retrieval Integration**   
On every user query, your conversation manager must: 

● Embed the query and retrieve the top-k most relevant chunks (k should be at least 3).   
● Inject the retrieved chunks into the prompt without exceeding the model's context window, you need an explicit  strategy for this. 

● Ground the response in the retrieved content, and refer back to the source document where it makes sense to. 

**Phase III — Keeping It Real-Time** 

Retrieval adds a step before the model can start generating, so it is the most likely place for this assignment to feel slower  than Assignment 1\. You must design around that: 

● **Latency:** benchmark retrieval time on its own, separate from generation time, and aim to keep it under one second.  Report the number in your README. The overall latency should be as minimum as possible. 

● **Streaming:** token streaming from the WebSocket endpoint must keep working exactly as it did in  Assignment 1\. 

● **Caching:** cache embeddings or results for queries you expect to repeat. 

● **Concurrency**: The system must support concurrent users as before. When multiple users trigger RAG or tools  simultaneously, the system should remain responsive. 

**Phase IV — Failure Handling** 

RAG introduces new ways for things to go wrong. Your system must handle each of these without crashing or hanging:  
● A query with no relevant matches in the corpus, the assistant should say so and fall back to normal conversation  rather than fabricating an answer from a weak match. 

● The vector store or embedding step failing or timing out.   
● A retrieved chunk that, combined with conversation history, would overflow the context window. **Evaluation Criteria** 

| Criterion Weight |  |
| :---- | :---- |
| **Correctness & Completeness**  | **50%** |
| **Viva (code walkthrough, live Q\&A, reproduce your prompts)**  | **25%** |
| **Real-time Behaviour & Performance (latency, streaming, failure  handling)**  | **25%** |
| **Total**  | **100%** |
| **Bonus — see the Bonus section below**  | **up to \+10%** |

**Deliverables & Submission Guidelines** 

Submit a GitHub repository (or .zip) containing: 

● Your document collection.   
● Your offline indexing pipeline script.   
● Your updated conversation manager with the retrieval module integrated.   
● An updated **README.md**: your embedding model and vector store choice and why, your chunking and top-k  parameters, your retrieval latency benchmarks, architecture diagram, and known limitations. 

**Bonus (up to \+10%)** 

Pick at most one for the bonus credit on this assignment: 

● **Hybrid search or re-ranking:** combine keyword and vector search, or add a re-ranking step over your top-k  results, and show whether it improves retrieval quality or not. 

● **Visible citations:** show the user which source document(s) an answer was grounded in, directly in the chat UI. ● **Something else:** If you have another creative idea, it will be rewarded with extra credit but it should not disrupt all the  required functionalities. 

**Honor Policy** 

Cheating in code, datasets, or reports will result in an immediate zero grade and will be reported to the academic  disciplinary committee. All work must be original to your group.