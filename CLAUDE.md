# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A LangChainGo (`github.com/tmc/langchaingo`) playground for building LLM agents in Go against the **Hydra** API — an OpenAI-compatible endpoint (`https://api.hydraai.ru/v1`), so the `llms/openai` client is used everywhere with a custom base URL. Code comments and seed data are in Russian; keep that convention when editing existing files.

## Two tiers of code

The repo has a deliberate split — understand which tier you're touching:

1. **`cmd/*_example/`** — standalone, self-contained learning scripts (`hydra_example`, `chain_example`, `agent_example`, `react_agent_example`, `rag_example`, `hydra_reasoning_example`). Each is its own `package main` with its own `main()`, shares **no** code, and **hardcodes the Hydra API key as a `const`**. These are demos/scratch, not production. `rag_example` implements an in-memory vector store with hand-rolled cosine similarity; `react_agent_example` has the original `CustomChain` ReAct loop.

2. **`pkg/reactrag/` + `cmd/react_rag_agent/`** — the productionized template distilled from the examples. This is the real deliverable: env-based config (no hardcoded keys), an HTTP service, Qdrant vector store, and tests. **New agent work should build on this tier, not the examples.**

When promoting logic from an example into `pkg/reactrag`, the pattern is: replace hardcoded keys with `Config`/`LoadFromEnv`, swap the in-memory store for Qdrant, and keep the English ReAct keywords intact (see below).

## Architecture of the `reactrag` template

The ReAct loop and RAG are decoupled — RAG is exposed to the agent as *just another tool*, so the LLM decides in its reasoning loop when to hit the knowledge base.

- [pkg/reactrag/agent.go](pkg/reactrag/agent.go) — `ReActAgent`. Runs a manual Thought/Action/Observation loop, parsing raw model output with **regexes** (`ParseModelResponse`). Uses `llms.WithStopWords(["Observation:"])` so the model can't hallucinate tool results. Accumulates history in a `scratchpad` string across iterations. Tool errors are returned *as observations* so the model can self-correct. `Run` returns the answer plus a `[]Step` trace.
- [pkg/reactrag/prompt.go](pkg/reactrag/prompt.go) — the ReAct prompt template. **The keywords (`Thought:`, `Action:`, `Action Input:`, `Observation:`, `Final Answer:`) are English on purpose — the regex parser in `agent.go` is keyed to them. Do not translate them.**
- [pkg/reactrag/rag.go](pkg/reactrag/rag.go) — `RAG` over PostgreSQL + pgvector. Unlike the Qdrant store, the langchaingo pgvector store self-initializes on `New` (creates the `vector` extension, the `langchain_pg_collection`/`langchain_pg_embedding` tables, and the collection), so there's no `EnsureCollection`. `NeedsSeeding` decides whether to seed by counting rows in the embedding table for the collection (uses the shared `pgxpool.Pool`). `Ingest` chunks docs (RecursiveCharacter, size 300 / overlap 30). `Search` formats hits into a text block usable as an Observation. Requires a Postgres image with pgvector (e.g. `pgvector/pgvector:pg17`).
- [pkg/reactrag/ragtool.go](pkg/reactrag/ragtool.go) — adapts `RAG.Search` to the langchaingo `tools.Tool` interface as `rag_search`.
- [pkg/reactrag/llm.go](pkg/reactrag/llm.go) — two separate `openai.New` clients: one for generation, one for embeddings.
- [pkg/reactrag/config.go](pkg/reactrag/config.go) — `LoadFromEnv`; `HYDRA_API_KEY` is the only required var.
- [pkg/reactrag/documents.go](pkg/reactrag/documents.go) — `SeedDocuments`, seeded into pgvector when the collection is empty.
- [cmd/react_rag_agent/](cmd/react_rag_agent/) — wires it all together behind a chi HTTP server. On boot: load config → build LLM+embedder → `NewRAG` (self-inits pgvector) → seed iff `NeedsSeeding` → register tools (`rag_search` + `tools.Calculator{}`) → serve `POST /api/agent/query`.

## Commands

```bash
# Run tests (parser/loop logic in pkg/reactrag; no network needed)
go test ./...
go test ./pkg/reactrag/ -run TestParseModelResponse -v   # single test

go vet ./...
go build ./...

# Run a standalone example (hardcodes its own key)
go run ./cmd/rag_example

# Run the full ReAct+RAG service
docker compose up -d postgres      # Postgres+pgvector on :5432
export HYDRA_API_KEY=...           # or use a .env file
go run ./cmd/react_rag_agent       # serves on :8080

# Or run everything (agent + postgres) in Docker
docker compose up --build          # agent Dockerfile at docker/Dockerfile

# Query the service
curl -X POST localhost:8080/api/agent/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"Что такое RAG и какие у него компоненты?"}'
```

## Config & secrets

- The `reactrag` template is env-configured — copy `.env.example` to `.env`. `HYDRA_API_KEY` is required; everything else has defaults (see [config.go](pkg/reactrag/config.go)). `.env` is gitignored.
- In `docker-compose.yml`, the agent reaches Postgres at `postgres:5432` (service name) via `DATABASE_URL`, overriding the `localhost` default.
- **The `cmd/*_example/` scripts contain a real-looking hardcoded Hydra key.** If you refactor or extend an example, do not propagate that pattern into `pkg/` — use `Config` instead.
