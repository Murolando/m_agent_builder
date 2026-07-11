package reactrag

import (
	"context"
	"fmt"
	"strings"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/textsplitter"
	"github.com/tmc/langchaingo/vectorstores/pgvector"
)

// RAG — обёртка над pgvector-хранилищем (PostgreSQL + расширение vector):
// инициализация таблиц/коллекции, засев документов и семантический поиск,
// отформатированный в текстовый контекст для агента.
type RAG struct {
	store          pgvector.Store
	pool           *pgxpool.Pool
	collectionName string
	topK           int
}

// NewRAG конструирует RAG поверх pgvector. В отличие от Qdrant, langchaingo
// pgvector-стор при инициализации сам создаёт расширение vector, таблицы
// коллекций/эмбеддингов и саму коллекцию — отдельный EnsureCollection не нужен.
func NewRAG(ctx context.Context, cfg RAGConfig, embedder embeddings.Embedder) (*RAG, error) {
	pool, err := pgxpool.New(ctx, cfg.DatabaseURL)
	if err != nil {
		return nil, fmt.Errorf("connect postgres: %w", err)
	}

	store, err := pgvector.New(ctx,
		pgvector.WithConn(pool),
		pgvector.WithEmbedder(embedder),
		pgvector.WithCollectionName(cfg.Collection),
		pgvector.WithVectorDimensions(cfg.VectorSize),
	)
	if err != nil {
		pool.Close()
		return nil, fmt.Errorf("init pgvector store: %w", err)
	}

	return &RAG{
		store:          store,
		pool:           pool,
		collectionName: cfg.Collection,
		topK:           cfg.TopK,
	}, nil
}

// Close закрывает пул соединений с PostgreSQL.
func (r *RAG) Close() {
	if r.pool != nil {
		r.pool.Close()
	}
}

// NeedsSeeding сообщает, пуста ли коллекция (в ней нет ни одного эмбеддинга) —
// то есть требуется засев стартовой базы знаний. Считаем строки в таблице
// эмбеддингов, связанные с нашей коллекцией по имени.
func (r *RAG) NeedsSeeding(ctx context.Context) (bool, error) {
	sql := fmt.Sprintf(
		`SELECT COUNT(*) FROM %s e
		 JOIN %s c ON e.collection_id = c.uuid
		 WHERE c.name = $1`,
		pgvector.DefaultEmbeddingStoreTableName,
		pgvector.DefaultCollectionStoreTableName,
	)

	var count int
	if err := r.pool.QueryRow(ctx, sql, r.collectionName).Scan(&count); err != nil {
		return false, fmt.Errorf("count embeddings: %w", err)
	}
	return count == 0, nil
}

// Ingest чанкит документы и загружает их в pgvector. Возвращает число чанков.
func (r *RAG) Ingest(ctx context.Context, docs []schema.Document) (int, error) {
	splitter := textsplitter.NewRecursiveCharacter()
	splitter.ChunkSize = 300
	splitter.ChunkOverlap = 30

	chunks, err := textsplitter.SplitDocuments(splitter, docs)
	if err != nil {
		return 0, fmt.Errorf("split documents: %w", err)
	}

	if _, err := r.store.AddDocuments(ctx, chunks); err != nil {
		return 0, fmt.Errorf("add documents to pgvector: %w", err)
	}
	return len(chunks), nil
}

// Search выполняет семантический поиск и форматирует найденное в текстовый
// контекст (как contextBuilder в rag_example), пригодный как Observation.
func (r *RAG) Search(ctx context.Context, query string) (string, error) {
	docs, err := r.store.SimilaritySearch(ctx, query, r.topK)
	if err != nil {
		return "", fmt.Errorf("similarity search: %w", err)
	}
	if len(docs) == 0 {
		return "No relevant information found in the knowledge base.", nil
	}

	var b strings.Builder
	b.WriteString("Relevant information from the knowledge base:\n\n")
	for i, doc := range docs {
		b.WriteString(fmt.Sprintf("%d. %s\n", i+1, strings.TrimSpace(doc.PageContent)))
		if src, ok := doc.Metadata["source"]; ok {
			b.WriteString(fmt.Sprintf("   Source: %v\n", src))
		}
		b.WriteString("\n")
	}
	return b.String(), nil
}
