package reactrag

import (
	"fmt"
	"strings"

	"github.com/tmc/langchaingo/tools"
)

// DefaultInstructions — настраиваемая «шапка» ReAct-промпта: роль агента и
// правила поведения. Именно её можно переопределить своим текстом (см.
// BuildPrompt). Формат ниже (Thought/Action/Observation) переопределять нельзя —
// под него заточен regex-парсер в protocol.go.
const DefaultInstructions = `You are a helpful assistant that answers questions about an internal knowledge base.
You solve problems step by step and rely on tools to gather facts instead of guessing.
If the knowledge base does not contain the answer, say so honestly in the Final Answer instead of making facts up.`

// reactTemplate — ReAct-промпт (перенос buildPrompt из react_agent_example).
// Первый %s — инструкции (кастомизируемая шапка), дальше идёт фиксированный
// формат. Ключевые слова английские — под них заточен regex-парсер в protocol.go.
const reactTemplate = `%s

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

Question: %s`

// BuildPrompt собирает полный ReAct-промпт: инструкции + описания инструментов +
// вопрос + накопленный scratchpad (история Thought/Action/Observation).
// Если instructions пустой, подставляется DefaultInstructions — так вызывающий
// код может передать свою «шапку», не трогая фиксированный ReAct-формат.
func BuildPrompt(instructions, question, scratchpad string, agentTools []tools.Tool) string {
	if strings.TrimSpace(instructions) == "" {
		instructions = DefaultInstructions
	}

	names := make([]string, 0, len(agentTools))
	descs := make([]string, 0, len(agentTools))
	for _, t := range agentTools {
		names = append(names, t.Name())
		descs = append(descs, fmt.Sprintf("%s: %s", t.Name(), t.Description()))
	}

	prompt := fmt.Sprintf(reactTemplate,
		instructions,
		strings.Join(descs, "\n"),
		strings.Join(names, ", "),
		question)

	if scratchpad != "" {
		prompt += "\n" + scratchpad
	}
	prompt += "\nThought:"
	return prompt
}
