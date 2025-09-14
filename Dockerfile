# Multi-stage build для оптимизации размера образа

# Стадия 1: Построение Go приложения
FROM golang:1.23.2-alpine AS go-builder

WORKDIR /app

# Устанавливаем зависимости для компиляции включая gcc для CGO
RUN apk add --no-cache git gcc musl-dev sqlite-dev

# Копируем go mod файлы и загружаем зависимости
COPY go.mod go.sum ./
RUN go mod download

# Копируем исходный код
COPY . .

# Компилируем приложение для glibc (не musl)
RUN CGO_ENABLED=1 GOOS=linux GOARCH=amd64 go build -tags netgo -ldflags '-extldflags "-static"' -o main .

# Стадия 2: Подготовка Python окружения
FROM python:3.11-slim AS python-builder

# Устанавливаем системные зависимости для OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем Python пакеты
RUN pip install --no-cache-dir \
    opencv-python==4.8.1.78 \
    pillow==10.0.1 \
    requests==2.31.0 \
    numpy==1.24.4 \
    onnxruntime==1.16.0

# Стадия 3: Финальный образ
FROM python:3.11-slim

# Устанавливаем системные зависимости для runtime
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Копируем Python пакеты из python-builder
COPY --from=python-builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=python-builder /usr/local/bin /usr/local/bin

WORKDIR /app

# Копируем скомпилированное Go приложение
COPY --from=go-builder /app/main /app/main

# Копируем необходимые файлы
COPY model/ ./model/
COPY static/ ./static/

# Создаем папки для данных
RUN mkdir -p uploads database

# Устанавливаем права на выполнение
RUN chmod +x /app/main

# Открываем порт
EXPOSE 8081

# Устанавливаем переменные окружения
ENV GIN_MODE=release

# Запускаем приложение
CMD ["/app/main"]