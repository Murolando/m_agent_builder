package reactrag

import (
	"fmt"
	"strings"

	"github.com/tmc/langchaingo/tools"
)

// reactTemplate — ReAct-промпт (перенос buildPrompt из react_agent_example).
// Ключевые слова английские — под них заточен regex-парсер в agent.go.
const reactTemplate = `You are a helpful assistant that answers questions about an internal knowledge base.
You solve problems step by step and rely on tools to gather facts instead of guessing.

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

If the knowledge base does not contain the answer, say so honestly in the Final Answer instead of making facts up.

Begin!

Question: %s`

// BuildPrompt собирает полный ReAct-промпт: шаблон + описания инструментов +
// вопрос + накопленный scratchpad (история Thought/Action/Observation).
func BuildPrompt(question, scratchpad string, agentTools []tools.Tool) string {
	names := make([]string, 0, len(agentTools))
	descs := make([]string, 0, len(agentTools))
	for _, t := range agentTools {
		names = append(names, t.Name())
		descs = append(descs, fmt.Sprintf("%s: %s", t.Name(), t.Description()))
	}

	prompt := fmt.Sprintf(reactTemplate,
		strings.Join(descs, "\n"),
		strings.Join(names, ", "),
		question)

	if scratchpad != "" {
		prompt += "\n" + scratchpad
	}
	prompt += "\nThought:"
	return prompt
}
