package repository

import "github.com/tmc/langchaingo/schema"

// SeedDocuments — стартовая база знаний для демонстрации (перенос sampleDocs
// из cmd/rag_example). Засевается в pgvector при первом запуске.
func SeedDocuments() []schema.Document {
	return []schema.Document{
		{
			PageContent: `RAG (Retrieval-Augmented Generation) - это технология, которая
			объединяет поиск информации с генерацией текста. RAG позволяет языковым
			моделям использовать внешние базы знаний для создания более точных и
			актуальных ответов. Основные компоненты RAG включают векторное хранилище,
			систему поиска и языковую модель. RAG решает проблему галлюцинаций LLM
			и обеспечивает актуальность информации.`,
			Metadata: map[string]any{"source": "rag_intro.txt", "topic": "основы"},
		},
		{
			PageContent: `Векторные базы данных хранят информацию в виде числовых векторов,
			что позволяет эффективно находить семантически похожие документы. Популярные
			векторные БД включают Pinecone, Weaviate, Chroma, Qdrant и FAISS. Каждая
			имеет свои преимущества: FAISS отлично подходит для больших объемов данных,
			Chroma проста в использовании, а Pinecone предоставляет managed service.
			Векторные БД используют косинусное сходство для поиска похожих документов.`,
			Metadata: map[string]any{"source": "vector_db.txt", "topic": "векторные БД"},
		},
		{
			PageContent: `LangChain и LangChainGo - это фреймворки для разработки приложений
			с использованием больших языковых моделей. Они предоставляют абстракции для
			работы с LLM, векторными хранилищами, памятью, агентами и инструментами.
			LangChainGo - это Go-версия популярного Python фреймворка LangChain.
			Основные компоненты включают chains, agents, prompts, memory и tools.`,
			Metadata: map[string]any{"source": "langchain.txt", "topic": "фреймворки"},
		},
		{
			PageContent: `Embeddings (эмбеддинги) - это числовые представления текста в
			многомерном пространстве. Они позволяют измерять семантическую близость
			между текстами. Популярные модели для создания embeddings включают
			text-embedding-3-small от OpenAI, text-embedding-ada-002, sentence-transformers
			и различные BERT-based модели. text-embedding-3-small - новая модель, которая
			более эффективна и дешевле предыдущих моделей.`,
			Metadata: map[string]any{"source": "embeddings.txt", "topic": "embeddings"},
		},
		{
			PageContent: `Чанкинг (chunking) - это процесс разбиения больших документов
			на меньшие фрагменты для обработки. Оптимальный размер чанка обычно
			составляет 200-1000 токенов с перекрытием 10-20%. Правильный чанкинг
			критически важен для качества поиска в RAG системах. Слишком большие
			чанки теряют точность, а слишком маленькие теряют контекст.`,
			Metadata: map[string]any{"source": "chunking.txt", "topic": "обработка"},
		},
		{
			PageContent: `OpenAI text-embedding-3-small - это новая модель эмбеддингов от OpenAI,
			представленная в 2024 году. Она имеет размерность 1536 и работает быстрее и
			дешевле, чем text-embedding-ada-002. Модель поддерживает настройку размерности
			через параметр dimensions, что позволяет оптимизировать производительность.
			Цена составляет примерно $0.02 за 1M токенов, что в 5 раз дешевле предыдущей модели.`,
			Metadata: map[string]any{"source": "openai_embeddings.txt", "topic": "модели"},
		},
	}
}
