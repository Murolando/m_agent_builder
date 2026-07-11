package reactrag

import (
	"fmt"

	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/openai"
)

// NewLLM создаёт генеративную модель Hydra (OpenAI-совместимый клиент).
func NewLLM(cfg HydraConfig) (llms.Model, error) {
	llm, err := openai.New(
		openai.WithToken(cfg.APIKey),
		openai.WithBaseURL(cfg.BaseURL),
		openai.WithModel(cfg.LLMModel),
	)
	if err != nil {
		return nil, fmt.Errorf("init hydra llm: %w", err)
	}
	return llm, nil
}

// NewEmbedder создаёт эмбеддер Hydra (отдельный клиент с embedding-моделью).
func NewEmbedder(cfg HydraConfig) (embeddings.Embedder, error) {
	embedLLM, err := openai.New(
		openai.WithToken(cfg.APIKey),
		openai.WithBaseURL(cfg.BaseURL),
		openai.WithEmbeddingModel(cfg.EmbedModel),
	)
	if err != nil {
		return nil, fmt.Errorf("init hydra embedding llm: %w", err)
	}

	embedder, err := embeddings.NewEmbedder(embedLLM)
	if err != nil {
		return nil, fmt.Errorf("create embedder: %w", err)
	}
	return embedder, nil
}
