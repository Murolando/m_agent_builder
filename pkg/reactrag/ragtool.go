package reactrag

import (
	"context"

	"github.com/tmc/langchaingo/tools"
)

// RagSearchTool делает RAG-поиск инструментом ReAct: агент в цикле рассуждений
// сам решает, когда сходить в базу знаний, вызвав действие "rag_search".
type RagSearchTool struct {
	rag *RAG
}

// статическая проверка соответствия интерфейсу tools.Tool.
var _ tools.Tool = (*RagSearchTool)(nil)

// NewRagSearchTool создаёт инструмент поиска по базе знаний.
func NewRagSearchTool(rag *RAG) *RagSearchTool {
	return &RagSearchTool{rag: rag}
}

func (t *RagSearchTool) Name() string {
	return "rag_search"
}

func (t *RagSearchTool) Description() string {
	return "Search the internal knowledge base for relevant information. " +
		"Use this whenever the question may be answered by stored documents. " +
		"Input: a natural-language search query."
}

func (t *RagSearchTool) Call(ctx context.Context, input string) (string, error) {
	return t.rag.Search(ctx, input)
}
