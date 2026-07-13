// Package agent — референс-оркестратор поверх ReAct-протокола из
// pkg/reactrag. Это ОДИН способ гонять цикл: простой линейный
// Thought/Action/Observation, один инструмент за шаг, накопление истории в
// scratchpad, без ветвления и параллелизма. Более сложная оркестрация (граф
// с узлами/переходами/state, суб-агенты, параллельные вызовы инструментов) —
// это альтернативный оркестратор, который строится на тех же примитивах
// протокола (reactrag.ParseModelResponse / ShouldContinue / BuildPrompt),
// не трогая их. Оркестратор держим в internal, а не в pkg, потому что это
// прикладная сборка, а не переиспользуемая библиотечная часть.
package agent

import (
	"context"
	"fmt"
	"strings"

	"github.com/Murolando/m_agent_builder/pkg/reactrag"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/tools"
)

// ReActAgent — линейный ReAct-агент. Парсит ответы модели протоколом reactrag,
// в цикле вызывает инструменты, собирает трейс шагов и возвращает финальный ответ.
type ReActAgent struct {
	llm   llms.Model
	tools map[string]tools.Tool
	// Не передаем их нативно через langchain, чтобы была возможность использовать разные модели
	toolList []tools.Tool
	maxSteps int
	verbose  bool
	// instructions — кастомная «шапка» ReAct-промпта (роль и правила поведения).
	// Пустая строка → reactrag.DefaultInstructions. Переопределить можно только
	// инструкции; фиксированный формат Thought/Action/Observation — нет.
	instructions string
}

// Option настраивает ReActAgent при создании. Функциональные опции позволяют
// расширять конфигурацию, не ломая существующие вызовы NewReActAgent.
type Option func(*ReActAgent)

// WithInstructions задаёт свою «шапку» промпта (роль и правила) вместо
// reactrag.DefaultInstructions. Пустая строка оставляет дефолт.
func WithInstructions(instructions string) Option {
	return func(a *ReActAgent) { a.instructions = instructions }
}

// NewReActAgent создаёт агента поверх модели и набора инструментов.
func NewReActAgent(llm llms.Model, agentTools []tools.Tool, maxSteps int, verbose bool, opts ...Option) *ReActAgent {
	toolMap := make(map[string]tools.Tool, len(agentTools))
	for _, t := range agentTools {
		toolMap[t.Name()] = t
	}
	if maxSteps <= 0 {
		maxSteps = 5
	}
	a := &ReActAgent{
		llm:      llm,
		tools:    toolMap,
		toolList: agentTools,
		maxSteps: maxSteps,
		verbose:  verbose,
	}
	for _, opt := range opts {
		opt(a)
	}
	return a
}

// Run выполняет ReAct-цикл и возвращает финальный ответ вместе с трейсом шагов.
func (a *ReActAgent) Run(ctx context.Context, question string) (answer string, steps []reactrag.Step, err error) {
	scratchpad := ""

	for step := 0; step < a.maxSteps; step++ {
		a.logf("\n=== Шаг %d ===\n", step+1)

		prompt := reactrag.BuildPrompt(a.instructions, question, scratchpad, a.toolList)

		response, err := llms.GenerateFromSinglePrompt(ctx, a.llm, prompt,
			// Останавливаемся на "Observation:", чтобы модель не дописывала
			// наблюдения за инструмент сама.
			llms.WithStopWords([]string{"Observation:"}),
		)
		if err != nil {
			return "", steps, fmt.Errorf("LLM call at step %d: %w", step+1, err)
		}
		a.logf("Ответ модели:\n%s\n", response)

		parsed := reactrag.ParseModelResponse(response)

		if parsed.FinalAnswer != "" {
			steps = append(steps, reactrag.Step{Thought: parsed.Thought})
			return parsed.FinalAnswer, steps, nil
		}

		if cont, reason := reactrag.ShouldContinue(parsed, step, a.maxSteps); !cont {
			// Ответ не распарсился, но попытки ещё есть — не падаем, а
			// возвращаем формат-подсказку как Observation, чтобы модель
			// исправилась на следующей итерации (в духе tool-error → observation).
			if reason == "no_clear_next_step" && step < a.maxSteps-1 {
				observation := "I couldn't parse your response. Reply strictly using " +
					"'Action:' and 'Action Input:' on separate lines, or 'Final Answer:'."
				a.logf("Observation (parse recovery): %s\n", observation)
				steps = append(steps, reactrag.Step{Observation: observation})
				scratchpad += strings.TrimRight(response, "\n") +
					fmt.Sprintf("\nObservation: %s\n", observation)
				continue
			}
			return "", steps, fmt.Errorf("agent stopped at step %d: %s\n%s", step+1, reason, response)
		}

		observation := a.executeTool(ctx, parsed.Action, parsed.ActionInput)
		a.logf("Observation: %s\n", observation)

		steps = append(steps, reactrag.Step{
			Thought:     parsed.Thought,
			Action:      parsed.Action,
			ActionInput: parsed.ActionInput,
			Observation: observation,
		})

		// Накапливаем цепочку рассуждений в scratchpad для следующей итерации.
		scratchpad += strings.TrimRight(response, "\n") +
			fmt.Sprintf("\nObservation: %s\n", observation)
	}

	return "", steps, fmt.Errorf("max iterations (%d) reached without final answer", a.maxSteps)
}

// executeTool вызывает инструмент по имени; ошибку отдаёт как observation,
// чтобы модель могла увидеть её и скорректировать действия.
func (a *ReActAgent) executeTool(ctx context.Context, name, input string) string {
	tool, ok := a.tools[name]
	if !ok {
		return fmt.Sprintf("Error: tool %q not found", name)
	}
	result, err := tool.Call(ctx, input)
	if err != nil {
		return fmt.Sprintf("Error: %s", err.Error())
	}
	return result
}

func (a *ReActAgent) logf(format string, args ...any) {
	if a.verbose {
		fmt.Printf(format, args...)
	}
}
