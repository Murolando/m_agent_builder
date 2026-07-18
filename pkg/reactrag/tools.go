package reactrag

// tools.go — НАТИВНЫЙ tool-calling оркестратор: второй режим библиотеки рядом с
// текстовым ReAct (BuildPrompt/ParseModelResponse/ShouldContinue из prompt.go и
// protocol.go). Разница — где проходит граница хода модели и среды:
//
//   - Текстовый ReAct: модель ПИШЕТ "Action:/Observation:" текстом, граница —
//     стоп-слово, наблюдение подставляется в общий текстовый поток. Портируется
//     на любую модель, но хрупко: если провайдер не уважает stop, модель
//     дописывает выдуманное Observation и финал за среду.
//   - Нативные тулы (здесь): модель эмитит СТРУКТУРНЫЙ tool_call, провайдер сам
//     останавливается по протоколу, наблюдение возвращается ОТДЕЛЬНЫМ tool-
//     сообщением. Модель физически не пишет наблюдение — выдумать его негде.
//     Требует, чтобы провайдер поддерживал OpenAI tools API.

import (
	"context"
	"fmt"

	"github.com/tmc/langchaingo/llms"
)

// NativeTool — инструмент для нативного цикла: схема (её видит модель) +
// исполнитель (твой код). Args в Call — сырой JSON-строка аргументов от модели.
type NativeTool struct {
	Name        string
	Description string
	// Parameters — JSON Schema аргументов, напр.
	//   map[string]any{"type":"object","properties":{...},"required":[...]}
	Parameters any
	Call       func(ctx context.Context, args string) (string, error)
}

// RunTools гоняет нативный tool-calling цикл поверх llms.Model:
//   - модель эмитит структурные tool_calls (не текст);
//   - цикл исполняет каждый и возвращает результат отдельным tool-сообщением;
//   - повторяет, пока модель не даст финальный текст (пустой tool_calls).
//
// Возвращает финальный ответ и трейс шагов (тот же Step, что и текстовый режим).
// Ни парсинга текста, ни стоп-слов, ни выдуманных Observation — граница хода
// структурная.
func RunTools(
	ctx context.Context,
	model llms.Model,
	systemPrompt, question string,
	tools []NativeTool,
	maxSteps int,
) (string, []Step, error) {
	if maxSteps <= 0 {
		maxSteps = 5
	}

	schemas := make([]llms.Tool, len(tools))
	byName := make(map[string]NativeTool, len(tools))
	for i, t := range tools {
		schemas[i] = llms.Tool{
			Type: "function",
			Function: &llms.FunctionDefinition{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.Parameters,
			},
		}
		byName[t.Name] = t
	}

	msgs := []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeSystem, systemPrompt),
		llms.TextParts(llms.ChatMessageTypeHuman, question),
	}

	var steps []Step
	for i := 0; i < maxSteps; i++ {
		resp, err := model.GenerateContent(ctx, msgs, llms.WithTools(schemas))
		if err != nil {
			return "", steps, fmt.Errorf("llm call at step %d: %w", i+1, err)
		}
		choice := resp.Choices[0]

		// Нет tool_calls — модель дала финальный ответ. Граница структурная:
		// «готово» сигнализируется отсутствием вызовов, а не текстом "Final Answer:".
		if len(choice.ToolCalls) == 0 {
			return choice.Content, steps, nil
		}

		// Ответ ассистента с его tool_calls — обратно в историю (как требует API).
		asst := llms.MessageContent{Role: llms.ChatMessageTypeAI}
		for _, tc := range choice.ToolCalls {
			asst.Parts = append(asst.Parts, tc)
		}
		msgs = append(msgs, asst)

		// Исполняем каждый вызов; результат — отдельным tool-сообщением по его ID.
		for _, tc := range choice.ToolCalls {
			if tc.FunctionCall == nil {
				continue
			}
			name, args := tc.FunctionCall.Name, tc.FunctionCall.Arguments

			var result string
			if tool, ok := byName[name]; ok {
				r, callErr := tool.Call(ctx, args)
				if callErr != nil {
					r = fmt.Sprintf("tool error: %v", callErr)
				}
				result = r
			} else {
				result = fmt.Sprintf("unknown tool: %s", name)
			}

			steps = append(steps, Step{Action: name, ActionInput: args, Observation: result})
			msgs = append(msgs, llms.MessageContent{
				Role: llms.ChatMessageTypeTool,
				Parts: []llms.ContentPart{
					llms.ToolCallResponse{ToolCallID: tc.ID, Name: name, Content: result},
				},
			})
		}
	}
	return "", steps, fmt.Errorf("max steps (%d) reached without final answer", maxSteps)
}
