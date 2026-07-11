# Этап 1: Сборка приложения
FROM golang:1.25.5-alpine AS builder

WORKDIR /app

# Копируем файлы зависимостей для кэширования слоёв
COPY go.mod go.sum ./
RUN go mod download

# Копируем исходный код
COPY . .

# Собираем ReAct+RAG агента со статической линковкой
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o agent ./cmd/react_rag_agent

# Этап 2: Финальный образ
FROM alpine:latest

WORKDIR /app

# Сертификаты нужны для HTTPS-запросов к Hydra API
RUN apk --no-cache add ca-certificates

# Копируем собранный бинарник из builder этапа
COPY --from=builder /app/agent .

EXPOSE 8080

CMD ["./agent"]
