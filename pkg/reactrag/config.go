package reactrag

import (
	"fmt"
	"os"
	"strconv"
)

// Конфигурация пакета намеренно разбита по компонентам: каждый конструктор
// берёт только тот конфиг, который ему реально нужен. Так пакет остаётся
// модульным — HTTP-порт и прочие детали приложения сюда не протекают, они
// живут в cmd. Каждый конфиг умеет грузиться из окружения (см. .env.example).

// HydraConfig — доступ к OpenAI-совместимому API Hydra (генерация + эмбеддинги).
// Используется NewLLM и NewEmbedder.
type HydraConfig struct {
	APIKey     string
	BaseURL    string
	LLMModel   string
	EmbedModel string
}

// RAGConfig — параметры хранилища и поиска (PostgreSQL + pgvector).
// Используется NewRAG.
type RAGConfig struct {
	// DatabaseURL — строка подключения к Postgres (postgres://...).
	DatabaseURL string
	// Collection — имя коллекции (логической группы эмбеддингов).
	Collection string
	// VectorSize — размерность эмбеддингов (text-embedding-3-small = 1536).
	VectorSize int
	// TopK — сколько документов возвращать при поиске.
	TopK int
}

// LoadHydraConfig собирает конфиг Hydra из окружения. Возвращает ошибку,
// если не задан обязательный HYDRA_API_KEY.
func LoadHydraConfig() (HydraConfig, error) {
	cfg := HydraConfig{
		APIKey:     os.Getenv("HYDRA_API_KEY"),
		BaseURL:    envOr("HYDRA_BASE_URL", "https://api.hydraai.ru/v1"),
		LLMModel:   envOr("HYDRA_LLM_MODEL", "gpt-4o-mini"),
		EmbedModel: envOr("HYDRA_EMBED_MODEL", "text-embedding-3-small"),
	}
	if cfg.APIKey == "" {
		return HydraConfig{}, fmt.Errorf("HYDRA_API_KEY is required")
	}
	return cfg, nil
}

func LoadRAGConfig() RAGConfig {
	return RAGConfig{
		DatabaseURL: envOr("DATABASE_URL", "postgres://postgres:postgres@localhost:5432/ragdb?sslmode=disable"),
		Collection:  envOr("PG_COLLECTION", "knowledge_base"),
		VectorSize:  envIntOr("VECTOR_SIZE", 1536),
		TopK:        envIntOr("RAG_TOP_K", 3),
	}
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func envIntOr(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}
