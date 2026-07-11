package main

import (
	"context"
	"fmt"
	"os"
	"regexp"
	"strings"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/openai"
	"github.com/tmc/langchaingo/tools"
)

// CustomChain - кастомная цепочка для обработки ответов
type CustomChain struct {
	llm      llms.Model
	tools    map[string]tools.Tool
	history  []string
	maxSteps int
}

// ParsedResponse - структура для разобранного ответа
type ParsedResponse struct {
	Thought     string
	Action      string
	ActionInput string
	FinalAnswer string
}

// NewCustomChain создает новую кастомную цепочку
func NewCustomChain(llm llms.Model, agentTools []tools.Tool) *CustomChain {
	toolMap := make(map[string]tools.Tool)
	for _, tool := range agentTools {
		toolMap[tool.Name()] = tool
	}

	return &CustomChain{
		llm:      llm,
		tools:    toolMap,
		history:  []string{},
		maxSteps: 5,
	}
}

// ParseModelResponse парсит ответ модели
func (c *CustomChain) ParseModelResponse(response string) *ParsedResponse {
	parsed := &ParsedResponse{}

	// Парсим компоненты ответа
	thoughtRegex := regexp.MustCompile(`Thought:\s*(.+?)(?:\n|$)`)
	actionRegex := regexp.MustCompile(`Action:\s*(.+?)(?:\n|$)`)
	actionInputRegex := regexp.MustCompile(`Action Input:\s*(.+?)(?:\n|$)`)
	finalAnswerRegex := regexp.MustCompile(`Final Answer:\s*(.+?)(?:\n|$)`)

	if match := thoughtRegex.FindStringSubmatch(response); len(match) > 1 {
		parsed.Thought = strings.TrimSpace(match[1])
	}

	if match := actionRegex.FindStringSubmatch(response); len(match) > 1 {
		parsed.Action = strings.TrimSpace(match[1])
	}

	if match := actionInputRegex.FindStringSubmatch(response); len(match) > 1 {
		parsed.ActionInput = strings.TrimSpace(match[1])
	}

	if match := finalAnswerRegex.FindStringSubmatch(response); len(match) > 1 {
		parsed.FinalAnswer = strings.TrimSpace(match[1])
	}

	return parsed
}

// ShouldContinue решает, нужно ли продолжать выполнение
func (c *CustomChain) ShouldContinue(parsed *ParsedResponse, step int) (bool, string) {
	// Если есть финальный ответ - завершаем
	if parsed.FinalAnswer != "" {
		return false, "final_answer"
	}

	// Если превышен лимит шагов
	if step >= c.maxSteps {
		return false, "max_steps_reached"
	}

	// Если есть действие для выполнения - продолжаем
	if parsed.Action != "" && parsed.ActionInput != "" {
		return true, "has_action"
	}

	// Если модель только думает, но не предпринимает действий
	if parsed.Thought != "" && parsed.Action == "" {
		return true, "thinking_only"
	}

	return false, "no_clear_next_step"
}

// ExecuteTool выполняет инструмент
func (c *CustomChain) ExecuteTool(ctx context.Context, toolName, input string) (string, error) {
	tool, exists := c.tools[toolName]
	if !exists {
		return "", fmt.Errorf("tool %s not found", toolName)
	}

	result, err := tool.Call(ctx, input)
	if err != nil {
		return "", err
	}

	return result, nil
}

// Run запускает цепочку
func (c *CustomChain) Run(ctx context.Context, question string) (string, error) {
	// Формируем начальный промпт
	prompt := c.buildPrompt(question)

	for step := 0; step < c.maxSteps; step++ {
		fmt.Printf("\n=== Шаг %d ===\n", step+1)

		// Вызываем модель
		response, err := llms.GenerateFromSinglePrompt(ctx, c.llm, prompt)
		if err != nil {
			return "", fmt.Errorf("ошибка вызова LLM: %w", err)
		}

		fmt.Printf("Ответ модели:\n%s\n", response)

		// Парсим ответ
		parsed := c.ParseModelResponse(response)

		// Добавляем в историю
		c.history = append(c.history, response)

		// Решаем, продолжать ли
		shouldContinue, reason := c.ShouldContinue(parsed, step)
		fmt.Printf("Решение: продолжить=%v, причина=%s\n", shouldContinue, reason)

		// Если есть финальный ответ - возвращаем его
		if parsed.FinalAnswer != "" {
			return parsed.FinalAnswer, nil
		}

		// Если не нужно продолжать
		if !shouldContinue {
			if reason == "max_steps_reached" {
				return "Достигнут лимит итераций", nil
			}
			break
		}

		// Выполняем действие, если оно есть
		if parsed.Action != "" && parsed.ActionInput != "" {
			observation, err := c.ExecuteTool(ctx, parsed.Action, parsed.ActionInput)
			if err != nil {
				observation = fmt.Sprintf("Error: %s", err.Error())
			}

			fmt.Printf("Результат действия: %s\n", observation)

			// Добавляем наблюдение в историю
			c.history = append(c.history, fmt.Sprintf("Observation: %s", observation))
		}

		// Обновляем промпт для следующей итерации
		prompt = c.buildPrompt(question)
	}

	return "Не удалось получить окончательный ответ", nil
}

// buildPrompt строит промпт с учетом истории
func (c *CustomChain) buildPrompt(question string) string {
	// Получаем описания инструментов
	var toolNames []string
	var toolDescriptions []string
	for name, tool := range c.tools {
		toolNames = append(toolNames, name)
		toolDescriptions = append(toolDescriptions,
			fmt.Sprintf("%s: %s", name, tool.Description()))
	}

	prompt := fmt.Sprintf(`You are a helpful assistant that solves problems step by step.

You have access to the following tools:
%s

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [%s]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: %s`,
		strings.Join(toolDescriptions, "\n"),
		strings.Join(toolNames, ", "),
		question)

	// Добавляем историю
	if len(c.history) > 0 {
		prompt += "\n" + strings.Join(c.history, "\n")
		prompt += "\nThought:"
	} else {
		prompt += "\nThought:"
	}

	return prompt
}

func main() {
	ctx := context.Background()

	apiKey := os.Getenv("HYDRA_API_KEY")
	if apiKey == "" {
		panic("не задана переменная окружения HYDRA_API_KEY")
	}

	// 1. Создаем LLM
	llm, err := openai.New(
		openai.WithToken(apiKey),
		openai.WithBaseURL("https://api.hydraai.ru/v1"),
		openai.WithModel("gpt-4o-mini"),
	)
	if err != nil {
		panic(err)
	}

	// 2. Создаем инструменты
	agentTools := []tools.Tool{
		tools.Calculator{},
	}

	// 3. Создаем кастомную цепочку
	chain := NewCustomChain(llm, agentTools)

	// 4. Запускаем с вопросом
	result, err := chain.Run(ctx, "What is 25 * 4 + 10?")
	if err != nil {
		panic(err)
	}

	fmt.Printf("\n=== ФИНАЛЬНЫЙ РЕЗУЛЬТАТ ===\n%s\n", result)
}
