# Assignment 3 — Voice and Tools

Due date / group rules: not included in the supplied excerpt.

## Introduction

Your assistant can already hold a conversation and ground its answers in your domain's documents. In this final assignment you will give it a voice and the ability to act: local speech-to-text and text-to-speech in a real-time pipeline, plus three tools the LLM can call. Everything from the earlier assignments (streaming, memory, RAG, concurrent users) must work the same way in voice mode.

## Objective

- Add local speech-to-text so users can speak to the assistant, with audio streamed to the backend.
- Add local text-to-speech that starts speaking before the full LLM response is ready.
- Implement three tools of your choice that the LLM can call, with timeouts and graceful error handling.
- Make streaming, memory, RAG, and tools work in voice mode exactly as they do in text mode.
- Keep latency close to real time with at least two concurrent users, running entirely locally.

## System architecture overview

You are extending the Assignment 2 architecture with a local voice pipeline (ASR and TTS) and a tool orchestrator alongside the retrieval module. The existing text WebSocket API contract must still not change; audio is added alongside it.

```text
Web UI <--WebSocket--> FastAPI Backend <--> Conversation Manager <--> Local LLM Engine
```

The Conversation Manager also connects to:

- Retrieval Module (Embedding + Vector Store)
- Voice Pipeline (local ASR + TTS)
- Tool Orchestrator -> Tool 1, Tool 2, Tool 3

## Tasks

### Phase I — Speech-to-Text

Let the user speak instead of typing, using a local ASR model (no cloud or browser speech APIs). It must:

- Stream microphone audio to the backend over the WebSocket in small chunks.
- Show the transcript, detect the end of speech, and produce a final transcript.
- Pass the final transcript to the same conversation manager used for typed text.

### Phase II — Text-to-Speech

Speak the assistant's replies using a local TTS model (no cloud services). It must:

- Start synthesizing sentence by sentence as LLM tokens stream, without waiting for the full response.
- Stream audio back to the browser in chunks and play it in order, while the text still appears in the chat.

### Phase III — Voice with Streaming, Memory, and RAG

Voice is not a separate system. Spoken turns must use the same LLM streaming, shared conversation history, and RAG pipeline as typed turns, and text-only chat must keep working.

### Phase IV — Tools

Implement three tools useful for your domain. Suggestions: weather (free API, e.g. OpenWeatherMap), calendar (local calendar file or Google Calendar API), calculator (evaluates math expressions), stock price (live or delayed), news summarisation (RSS headlines), database query (product catalogue or local SQLite). You may substitute your own tools entirely as long as they are genuinely useful for the business.

- Each tool needs a clear name, a description, and a JSON schema for its arguments.
- The orchestrator must detect when a tool is needed, call it, and pass the result back to the LLM for the final answer (prompt-based function calling or a library; be ready to explain it in the viva).
- Tool calls must be asynchronous, with a timeout, and failures must produce a sensible message, not a hang or crash.
- Tools must work from spoken requests, and their results must be spoken back.

### Phase V — Real-Time Behaviour

Measure and report per voice turn: ASR latency, retrieval/tool latency, LLM first-token latency, and TTS first-audio latency. The target is close to real time: ideally around one second and no more than about two seconds from end of speech to first audio (retrieval and tool calls together should add no more than about two seconds before generation starts). The system must handle at least two concurrent users, text or voice, without one blocking another.

## Evaluation

| Criterion | Weight |
| --- | ---: |
| Correctness & Completeness | 50% |
| Viva (code walkthrough, live Q&A, reproduce your prompts, voice pipeline, tool relevance to chosen domain) | 25% |
| Real-time Behaviour & Performance (latency, streaming, failure handling) | 25% |
| Bonus | up to +10% |

## Deliverables

Single private GitHub repo or .zip containing:

- The voice pipeline code (browser audio capture and playback, local ASR and TTS modules, streaming orchestration) and setup instructions for the local models.
- The tool orchestrator code and all three tool implementations, each with its schema documented.
- A final `README.md` covering the complete three assignments: an up-to-date architecture diagram, your model choices (LLM, ASR, TTS), your document collection and retrieval setup, each tool's name, description, schema, and an example of the LLM invoking it, your latency benchmarks (ASR, retrieval, tools, LLM, TTS, and end of speech to first audio), setup instructions, and known limitations.

## Bonus (up to +10%)

A CRM tool that remembers users across sessions. It must be exposed as something the LLM can call, with a clear name, description, and argument schema. At minimum it must:

- Store user information (name, contact details, preferences, past interactions) keyed by a session or user ID.
- Retrieve that information at the start of a returning session, so the assistant can greet the user by name and recall what they asked for last time.
- Update the record during the conversation when the user provides new information.

A simple persisted key-value store (a JSON file or SQLite) is enough. The important part is that the LLM calls it correctly, not how fancy the storage layer is. The CRM tool must also work in voice mode.
