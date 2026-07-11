// Command react_rag_agent поднимает HTTP-сервис ReAct-агента, у которого
// поиск по базе знаний (PostgreSQL + pgvector) — это инструмент rag_search.
// Агент в цикле рассуждений сам решает, когда обратиться к базе знаний.
//
// Запуск:
//
//	docker-compose up -d postgres
//	export HYDRA_API_KEY=...
//	go run ./cmd/react_rag_agent
//
// Запрос:
//
//	curl -X POST localhost:8080/api/agent/query \
//	  -H 'Content-Type: application/json' \
//	  -d '{"question":"Что такое RAG и какие у него компоненты?"}'
package main

import (
	"context"
	"log/slog"
	"net/http"
	"os"

	"github.com/Murolando/m_agent_builder/internal/agent"
	"github.com/Murolando/m_agent_builder/internal/repository"
	"github.com/Murolando/m_agent_builder/pkg/reactrag"
	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"github.com/tmc/langchaingo/tools"
)

func main() {
	log := slog.New(slog.NewTextHandler(os.Stdout, nil))
	ctx := context.Background()

	// 1. Компонентные конфиги из окружения (каждый — только про свой слой).
	hydraCfg, err := reactrag.LoadHydraConfig()
	if err != nil {
		log.Error("load config", "err", err)
		os.Exit(1)
	}
	ragCfg := reactrag.LoadRAGConfig()

	// 2. LLM + embedder (Hydra).
	llm, err := reactrag.NewLLM(hydraCfg)
	if err != nil {
		log.Error("init llm", "err", err)
		os.Exit(1)
	}
	embedder, err := reactrag.NewEmbedder(hydraCfg)
	if err != nil {
		log.Error("init embedder", "err", err)
		os.Exit(1)
	}

	// 3. RAG поверх pgvector: стор сам создаёт расширение/таблицы/коллекцию,
	// а мы при пустой коллекции засеваем стартовую базу знаний.
	rag, err := reactrag.NewRAG(ctx, ragCfg, embedder)
	if err != nil {
		log.Error("init rag", "err", err)
		os.Exit(1)
	}
	defer rag.Close()

	needsSeeding, err := rag.NeedsSeeding(ctx)
	if err != nil {
		log.Error("check knowledge base", "err", err)
		os.Exit(1)
	}
	if needsSeeding {
		log.Info("knowledge base is empty, seeding", "collection", ragCfg.Collection)
		n, err := rag.Ingest(ctx, repository.SeedDocuments())
		if err != nil {
			log.Error("seed knowledge base", "err", err)
			os.Exit(1)
		}
		log.Info("ingested chunks", "count", n)
	} else {
		log.Info("knowledge base already populated, skipping seed", "collection", ragCfg.Collection)
	}

	// 4. ReAct-агент: rag_search + калькулятор.
	agentTools := []tools.Tool{
		reactrag.NewRagSearchTool(rag),
		tools.Calculator{},
	}
	reactAgent := agent.NewReActAgent(llm, agentTools, 5, true)

	// 5. HTTP-сервер на chi.
	handler := NewAgentHandler(reactAgent, log)
	r := chi.NewRouter()
	r.Use(middleware.RequestID)
	r.Use(middleware.Recoverer)
	r.Route("/api", func(r chi.Router) {
		r.Post("/agent/query", handler.Query)
	})

	// Порт сервера — деталь транспорта, читаем прямо здесь, а не в пакете.
	port := os.Getenv("SERVER_PORT")
	if port == "" {
		port = "8080"
	}
	addr := ":" + port
	log.Info("listening", "addr", addr)
	if err := http.ListenAndServe(addr, r); err != nil {
		log.Error("server stopped", "err", err)
		os.Exit(1)
	}
}
