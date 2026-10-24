# Repository Guidelines

## Project Structure & Module Organization

- `backend/main.py` defines FastAPI REST endpoints, WebSocket chat, and frontend serving.
- `backend/retrieval/` owns CPU embeddings, persistent Chroma indexing, bounded retrieval, and caching; `data/corpus/` contains 60 source documents.
- `backend/conversation_manager/` handles intent, sessions, prompt budgets, and summaries; separate modules manage inference and persistence.
- `frontend/` uses plain HTML/CSS/JavaScript without a build step.
- `data/` holds fictional fixtures; runtime conversation history is ignored.
- `scripts/` provides setup and validation; `docs/superpowers/` stores plans.
- `tests/` covers scope, retrieval, memory, and streaming.

## Build, Test, and Development Commands

Run commands from the repository root with Python 3.13 (the validated environment):

- `python -m venv .venv`: create an isolated environment; activate it with `.venv/Scripts/Activate.ps1` on PowerShell.
- `python -m pip install -r requirements-lock.txt`: install validated dependencies.
- `ollama serve`: start local inference if Ollama is not already running.
- `ollama pull qwen2.5:1.5b`: download the default model.
- `uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload`: run development mode; open `http://localhost:8000`.
- `python scripts/seed_data_check.py`: validate mock-data fields and policy sections.
- `python scripts/prepare_models.py`: download local embedding weights and tokenizer once.
- `python scripts/validate_corpus.py`: validate document metadata and fixture facts.
- `python scripts/index_documents.py`: update the index with the backend stopped; `--rebuild` handles model changes.
- `python -m pytest`: run scope, indexing, retrieval, memory, budget, and WebSocket regression tests.

## Coding Style & Naming Conventions

Use four-space Python indentation, snake_case functions/modules, PascalCase classes, and UPPER_CASE constants. Preserve type hints and async streaming interfaces. Match nearby frontend formatting; JavaScript uses camelCase and two-space indentation. No formatter or linter is configured.

## Testing Guidelines

Pytest and pytest-asyncio are pinned dependencies; no coverage threshold is configured. Add tests under `tests/test_<module>.py`, with functions named `test_<behavior>`. Mock Ollama for deterministic unit tests. Cover intent switching, context handling, data lookup, and session/history behavior. For UI changes, manually check streaming, session reset, history resume/deletion, and narrow viewports.

## Commit & Pull Request Guidelines

Recent commits use imperative, descriptive subjects such as “Add REST endpoints…” and “Fix sidebar clipping…”. Follow that style and keep commits focused. PRs should explain the behavior change, list validation performed, link relevant issues, and include screenshots for UI changes.

## Security & Configuration Tips

Use `.env.example` as a configuration reference; supply environment variables or Uvicorn's `--env-file .env` option explicitly. Never commit secrets, conversation history, logs, or model weights. Keep fixture customer data fictional and inference local and CPU-only.
