package agent

import (
	"context"
	"fmt"
	"testing"

	"github.com/tmc/langchaingo/llms"
)

// scriptedLLM — фейковая модель, отдающая заранее заданные ответы по очереди.
// Позволяет проверять оркестратор без сети.
type scriptedLLM struct {
	responses []string
	calls     int
}

func (s *scriptedLLM) GenerateContent(_ context.Context, _ []llms.MessageContent, _ ...llms.CallOption) (*llms.ContentResponse, error) {
	if s.calls >= len(s.responses) {
		return nil, fmt.Errorf("scriptedLLM: no more responses (call %d)", s.calls+1)
	}
	r := s.responses[s.calls]
	s.calls++
	return &llms.ContentResponse{Choices: []*llms.ContentChoice{{Content: r}}}, nil
}

func (s *scriptedLLM) Call(_ context.Context, _ string, _ ...llms.CallOption) (string, error) {
	return "", fmt.Errorf("not implemented")
}

// Непарсящийся ответ не роняет запрос: цикл возвращает формат-подсказку как
// Observation и даёт модели исправиться на следующем шаге.
func TestRun_RecoversFromUnparseable(t *testing.T) {
	llm := &scriptedLLM{responses: []string{
		"I have no idea what format to use.", // мусор → самокоррекция
		"Thought: I know it\nFinal Answer: RAG is retrieval-augmented generation.",
	}}
	a := NewReActAgent(llm, nil, 5, false)

	answer, steps, err := a.Run(context.Background(), "what is RAG?")
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}
	if answer != "RAG is retrieval-augmented generation." {
		t.Errorf("answer = %q", answer)
	}
	if llm.calls != 2 {
		t.Errorf("expected 2 LLM calls (garbage + recovery), got %d", llm.calls)
	}
	// Первый шаг — восстановительное наблюдение.
	if len(steps) == 0 || steps[0].Observation == "" {
		t.Errorf("expected a recovery observation step, got %+v", steps)
	}
}

// Если попыток не осталось, непарсящийся ответ — это честная ошибка, а не
// бесконечный цикл.
func TestRun_UnparseableNoBudgetErrors(t *testing.T) {
	llm := &scriptedLLM{responses: []string{"garbage with no keywords"}}
	a := NewReActAgent(llm, nil, 1, false)

	if _, _, err := a.Run(context.Background(), "q"); err == nil {
		t.Fatal("expected error when unparseable and no steps left")
	}
}
