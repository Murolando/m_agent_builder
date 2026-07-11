package reactrag

import (
	"regexp"
	"strings"
)

// protocol.go — чистое ядро ReAct: как разобрать ответ модели и как решить,
// продолжать ли цикл. Здесь нет ни LLM, ни инструментов, ни цикла — только
// чистые функции и типы. Это переиспользуемый слой: любой оркестратор
// (линейный Run в agent.go или, в перспективе, граф) строится поверх него.
// Формат промпта — вторая половина протокола, см. prompt.go. Ключевые слова
// (Thought/Action/Action Input/Observation/Final Answer) английские намеренно:
// под них заточены регулярки ниже.

// Step — один разобранный шаг ReAct-цикла. Возвращается в трейсе HTTP-ответа.
type Step struct {
	Thought     string `json:"thought,omitempty"`
	Action      string `json:"action,omitempty"`
	ActionInput string `json:"action_input,omitempty"`
	Observation string `json:"observation,omitempty"`
}

// ParsedResponse — результат разбора одного ответа модели.
type ParsedResponse struct {
	Thought     string
	Action      string
	ActionInput string
	FinalAnswer string
}

// Регулярки толерантны к тому, что модели делают на практике:
//   - (?i) — регистр меток не важен (Action / action / ACTION);
//   - ^ с (?m) на строчных полях — метку ловим в начале строки, чтобы не
//     цепляться за упоминание в прозе;
//   - [ \t>#*-]* — допускаем markdown-маркеры/буллеты перед меткой;
//   - \s*: — пробел перед двоеточием ("Action :") не ломает разбор.
// Action Input и Final Answer разбираем dotall-режимом (?s), чтобы не терять
// многострочный ввод (JSON и т.п.); терминатор — \z (конец текста), а не $,
// иначе в многострочном режиме захват оборвётся на первой же строке.
var (
	thoughtRegex     = regexp.MustCompile(`(?im)^[ \t>#*-]*Thought\s*:\s*(.+?)\s*$`)
	actionRegex      = regexp.MustCompile(`(?im)^[ \t>#*-]*Action\s*:\s*(.+?)\s*$`)
	actionInputRegex = regexp.MustCompile(`(?is)Action\s+Input\s*:\s*(.+?)(?:\n\s*Observation\s*:|\n\s*Thought\s*:|\z)`)
	finalAnswerRegex = regexp.MustCompile(`(?is)Final\s+Answer\s*:\s*(.+)`)
)

// ParseModelResponse разбирает сырой ответ модели на компоненты ReAct.
// Перед разбором нормализует markdown-обёртки вокруг ключевых слов.
func ParseModelResponse(response string) ParsedResponse {
	response = normalizeResponse(response)

	var p ParsedResponse
	if m := thoughtRegex.FindStringSubmatch(response); len(m) > 1 {
		p.Thought = strings.TrimSpace(m[1])
	}
	if m := finalAnswerRegex.FindStringSubmatch(response); len(m) > 1 {
		p.FinalAnswer = strings.TrimSpace(m[1])
		// Final Answer имеет приоритет: действие игнорируем.
		return p
	}
	if m := actionRegex.FindStringSubmatch(response); len(m) > 1 {
		// Симметрично с ActionInput срезаем кавычки/бэктики/точку, иначе
		// имя тула ("rag_search") не совпадёт с зарегистрированным.
		p.Action = strings.Trim(strings.TrimSpace(m[1]), " \t\"'`.")
	}
	if m := actionInputRegex.FindStringSubmatch(response); len(m) > 1 {
		input := stripCodeFence(strings.TrimSpace(m[1]))
		p.ActionInput = strings.Trim(input, "\"`")
	}
	return p
}

// normalizeResponse снимает markdown-обёртки (**Action:**, __Action:__),
// мешающие сматчить ключевые слова.
func normalizeResponse(s string) string {
	return strings.NewReplacer("**", "", "__", "").Replace(s)
}

// stripCodeFence разворачивает Action Input, если модель обернула его в
// ```-блок (```json { ... } ```), возвращая только содержимое.
func stripCodeFence(s string) string {
	s = strings.TrimSpace(s)
	if !strings.HasPrefix(s, "```") {
		return s
	}
	// Убираем открывающую строку ```lang и закрывающие ```.
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		s = s[i+1:]
	} else {
		s = strings.TrimPrefix(s, "```")
	}
	s = strings.TrimSuffix(strings.TrimSpace(s), "```")
	return strings.TrimSpace(s)
}

// ShouldContinue решает, продолжать ли цикл, и поясняет причину.
func ShouldContinue(p ParsedResponse, step, maxSteps int) (bool, string) {
	if p.FinalAnswer != "" {
		return false, "final_answer"
	}
	if step >= maxSteps {
		return false, "max_steps_reached"
	}
	if p.Action != "" && p.ActionInput != "" {
		return true, "has_action"
	}
	if p.Thought != "" && p.Action == "" {
		return true, "thinking_only"
	}
	return false, "no_clear_next_step"
}
