# Настройка стабильности результатов

## Проблема разных результатов

Система может выдавать разные результаты по следующим причинам:

### 1. Цепочка fallback моделей
```
1. Roboflow API (если есть API ключ)
2. Offline Analyzer (если Roboflow не работает)
3. ONNX Model (если offline не работает)
4. Mock Analyzer (последний резерв)
```

### 2. Переключение между моделями
- Если Roboflow API недоступен → используется offline анализатор
- Разные модели дают разные результаты

## Решения для стабильности

### Опция 1: Принудительно использовать только Roboflow
Отключить fallback модели в коде:

```go
// Только Roboflow, без fallback
if apiKey := os.Getenv("ROBOFLOW_API_KEY"); apiKey != "" {
    modelResponse, err = callRoboflowModel(filepath)
    if err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": "Roboflow API unavailable"})
        return
    }
}
```

### Опция 2: Принудительно использовать только Offline
Убрать проверку API ключа:

```go
// Только offline analyzer
modelResponse, err := callOfflineModel(filepath)
```

### Опция 3: Кеширование результатов
Сохранять результаты по хешу изображения.

## Настройки Roboflow для стабильности

Увеличены пороги уверенности:
- `confidence: 30` (вместо 25)
- `overlap: 20` (вместо 30)

## Логирование

Теперь система показывает какая модель используется в консоли сервера.