package reactrag

import "testing"

func TestParseModelResponse_Action(t *testing.T) {
	resp := "Thought: I should look this up\n" +
		"Action: rag_search\n" +
		"Action Input: what is RAG\n"

	p := ParseModelResponse(resp)

	if p.Thought != "I should look this up" {
		t.Errorf("Thought = %q", p.Thought)
	}
	if p.Action != "rag_search" {
		t.Errorf("Action = %q", p.Action)
	}
	if p.ActionInput != "what is RAG" {
		t.Errorf("ActionInput = %q", p.ActionInput)
	}
	if p.FinalAnswer != "" {
		t.Errorf("FinalAnswer should be empty, got %q", p.FinalAnswer)
	}
}

func TestParseModelResponse_FinalAnswerWins(t *testing.T) {
	// Даже если в тексте мелькает Action, Final Answer имеет приоритет.
	resp := "Thought: I now know the answer\n" +
		"Action: rag_search\n" +
		"Final Answer: RAG combines retrieval with generation.\n"

	p := ParseModelResponse(resp)

	if p.FinalAnswer != "RAG combines retrieval with generation." {
		t.Errorf("FinalAnswer = %q", p.FinalAnswer)
	}
	if p.Action != "" {
		t.Errorf("Action should be ignored when Final Answer present, got %q", p.Action)
	}
}

func TestParseModelResponse_MultilineActionInput(t *testing.T) {
	resp := "Thought: build a query\n" +
		"Action: rag_search\n" +
		"Action Input: {\n  \"q\": \"vector db\"\n}\n" +
		"Observation: ignored"

	p := ParseModelResponse(resp)

	want := "{\n  \"q\": \"vector db\"\n}"
	if p.ActionInput != want {
		t.Errorf("ActionInput = %q, want %q", p.ActionInput, want)
	}
}

func TestParseModelResponse_Dirty(t *testing.T) {
	cases := []struct {
		name       string
		resp       string
		wantAction string
		wantInput  string
	}{
		{
			name:       "quoted action name",
			resp:       "Thought: look it up\nAction: \"rag_search\"\nAction Input: what is RAG\n",
			wantAction: "rag_search",
			wantInput:  "what is RAG",
		},
		{
			name:       "trailing dot on action",
			resp:       "Action: rag_search.\nAction Input: vector db\n",
			wantAction: "rag_search",
			wantInput:  "vector db",
		},
		{
			name:       "markdown bold keywords",
			resp:       "**Thought:** hmm\n**Action:** rag_search\n**Action Input:** what is RAG\n",
			wantAction: "rag_search",
			wantInput:  "what is RAG",
		},
		{
			name:       "lowercase keywords",
			resp:       "thought: hmm\naction: rag_search\naction input: what is RAG\n",
			wantAction: "rag_search",
			wantInput:  "what is RAG",
		},
		{
			name:       "space before colon",
			resp:       "Action : rag_search\nAction Input : what is RAG\n",
			wantAction: "rag_search",
			wantInput:  "what is RAG",
		},
		{
			name:       "code-fenced action input",
			resp:       "Action: rag_search\nAction Input:\n```json\n{\"q\": \"vector db\"}\n```\n",
			wantAction: "rag_search",
			wantInput:  "{\"q\": \"vector db\"}",
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			p := ParseModelResponse(c.resp)
			if p.Action != c.wantAction {
				t.Errorf("Action = %q, want %q", p.Action, c.wantAction)
			}
			if p.ActionInput != c.wantInput {
				t.Errorf("ActionInput = %q, want %q", p.ActionInput, c.wantInput)
			}
		})
	}
}

func TestParseModelResponse_Unparseable(t *testing.T) {
	// Мусор без ключевых слов — все поля пустые, а ShouldContinue должен
	// сказать "нет ясного следующего шага" (сигнал для самокоррекции цикла).
	p := ParseModelResponse("I think the answer is probably RAG but I'm not sure.")
	if p.Action != "" || p.ActionInput != "" || p.FinalAnswer != "" {
		t.Fatalf("expected empty parse, got %+v", p)
	}
	if cont, reason := ShouldContinue(p, 0, 5); cont || reason != "no_clear_next_step" {
		t.Errorf("ShouldContinue = (%v, %q), want (false, no_clear_next_step)", cont, reason)
	}
}

func TestShouldContinue(t *testing.T) {
	cases := []struct {
		name     string
		p        ParsedResponse
		step     int
		maxSteps int
		want     bool
		reason   string
	}{
		{"final answer stops", ParsedResponse{FinalAnswer: "done"}, 0, 5, false, "final_answer"},
		{"has action continues", ParsedResponse{Action: "rag_search", ActionInput: "q"}, 0, 5, true, "has_action"},
		{"max steps stops", ParsedResponse{Action: "rag_search", ActionInput: "q"}, 5, 5, false, "max_steps_reached"},
		{"thinking only continues", ParsedResponse{Thought: "hmm"}, 0, 5, true, "thinking_only"},
		{"empty stops", ParsedResponse{}, 0, 5, false, "no_clear_next_step"},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			cont, reason := ShouldContinue(c.p, c.step, c.maxSteps)
			if cont != c.want || reason != c.reason {
				t.Errorf("ShouldContinue = (%v, %q), want (%v, %q)", cont, reason, c.want, c.reason)
			}
		})
	}
}
