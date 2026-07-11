package main

import (
	"encoding/json"
	"log/slog"
	"net/http"

	"github.com/Murolando/m_agent_builder/internal/agent"
	"github.com/Murolando/m_agent_builder/pkg/reactrag"
)

// AgentHandler обслуживает POST /api/agent/query поверх ReAct+RAG агента.
type AgentHandler struct {
	agent *agent.ReActAgent
	log   *slog.Logger
}

func NewAgentHandler(agent *agent.ReActAgent, log *slog.Logger) *AgentHandler {
	return &AgentHandler{agent: agent, log: log}
}

type queryRequest struct {
	Question string `json:"question"`
}

type queryResponse struct {
	Answer string          `json:"answer"`
	Steps  []reactrag.Step `json:"steps"`
}

type errorResponse struct {
	Error string `json:"error"`
}

// Query — обработчик POST /api/agent/query.
func (h *AgentHandler) Query(w http.ResponseWriter, r *http.Request) {
	var req queryRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, errorResponse{Error: "invalid JSON body"})
		return
	}
	if req.Question == "" {
		writeJSON(w, http.StatusBadRequest, errorResponse{Error: "field 'question' is required"})
		return
	}

	h.log.Info("agent query", "question", req.Question)

	answer, steps, err := h.agent.Run(r.Context(), req.Question)
	if err != nil {
		h.log.Error("agent run failed", "err", err)
		writeJSON(w, http.StatusInternalServerError, errorResponse{Error: err.Error()})
		return
	}

	writeJSON(w, http.StatusOK, queryResponse{Answer: answer, Steps: steps})
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}
