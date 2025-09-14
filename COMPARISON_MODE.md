# Режим сравнения моделей - Roboflow + Offline

## 🔄 **Новый подход**

Система теперь запускает **оба метода анализа параллельно** и сравнивает результаты:

1. **🤖 Roboflow API** (4 специализированные модели)
2. **🔬 Offline Analyzer** (компьютерное зрение OpenCV)

## ⚡ **Параллельное выполнение**

```go
// Запускаем одновременно в goroutines
go func() { roboflowResponse = callRoboflowModel(filepath) }()
go func() { offlineResponse = callOfflineModel(filepath) }()
```

**Преимущества:**
- Быстрее чем последовательное выполнение
- Надежность - если один метод не работает, есть второй
- Научное сравнение результатов

## 📊 **Что получаешь в результате**

### **1. Индивидуальные результаты**
```json
{
  "model_comparison": {
    "roboflow": {
      "rust": true,
      "scratches": false,
      "method": "roboflow",
      "damage_details": [...]
    },
    "offline": {
      "rust": false,
      "scratches": true,
      "method": "offline",
      "damage_details": [...]
    }
  }
}
```

### **2. Анализ согласованности**
```json
{
  "agreement": {
    "rust": false,      // Методы не согласны
    "scratches": false, // Методы не согласны
    "dirt": true,       // Методы согласны
    "cracks": true,     // Методы согласны
    "dents": true       // Методы согласны
  },
  "conflicts": [
    "rust: Roboflow=true, Offline=false",
    "scratches: Roboflow=false, Offline=true"
  ]
}
```

### **3. Объединенный результат**
```json
{
  "combined": {
    "rust": true,        // Если ЛЮБОЙ метод обнаружил - включаем
    "scratches": true,   // Объединяем обнаружения
    "cleanliness": 0.75, // Взвешенное среднее (70% Roboflow + 30% Offline)
    "method": "combined (agreement: 3/5)",
    "damage_details": [...], // Объединяем все найденные дефекты
    "description": "Issues found by combined analysis: [rust, scratches]"
  }
}
```

## 🎯 **Логика объединения**

### **Веса методов:**
- **Roboflow**: 70% (обученные модели)
- **Offline**: 30% (эвристические алгоритмы)

### **Если методы сильно согласны (4+/5):**
- **Roboflow**: 60%
- **Offline**: 40% (больше доверия offline методу)

### **Стратегия обнаружения:**
- **OR логика**: Если ЛЮБОЙ метод нашел дефект → включаем
- **Максимальная уверенность**: Берем наибольшую confidence
- **Комбинация деталей**: Объединяем все найденные области

## 🔄 **Сценарии работы**

### **✅ Оба работают:**
- Получаем полное сравнение
- Взвешенное объединение результатов
- Анализ согласованности

### **⚠️ Только Roboflow работает:**
```json
{
  "method": "roboflow (offline failed)",
  "description": "Analysis based on Roboflow models only"
}
```

### **⚠️ Только Offline работает:**
```json
{
  "method": "offline (roboflow failed)",
  "description": "Analysis based on computer vision only"
}
```

### **❌ Оба не работают:**
```json
{
  "error": "Both analysis methods failed",
  "roboflow_error": "...",
  "offline_error": "..."
}
```

## 🎨 **Пример полного ответа**

```json
{
  "id": "analysis-123",
  "has_rust": true,
  "has_scratches": true,
  "overall_status": "Needs Attention",
  "details": "Issues found by combined analysis: [rust, scratches]",

  "model_comparison": {
    "roboflow": {
      "rust": true,
      "scratches": false,
      "confidence_scores": {"rust": 0.85},
      "damage_details": [
        {
          "type": "rust",
          "x": 245, "y": 167,
          "confidence": 0.85,
          "detected_by": "Rust Detection (SPECIALIZED for rust only)"
        }
      ]
    },
    "offline": {
      "rust": false,
      "scratches": true,
      "confidence_scores": {"scratches": 0.6},
      "damage_details": [
        {
          "type": "scratch",
          "x": 300, "y": 200,
          "confidence": 0.6,
          "detected_by": "OpenCV Edge Detection"
        }
      ]
    },
    "agreement": {
      "rust": false,
      "scratches": false,
      "dirt": true,
      "cracks": true,
      "dents": true
    },
    "conflicts": [
      "rust: Roboflow=true, Offline=false",
      "scratches: Roboflow=false, Offline=true"
    ],
    "combined": {
      "rust": true,
      "scratches": true,
      "method": "combined (agreement: 3/5)",
      "damage_details": [
        // Все дефекты от обоих методов
      ]
    }
  }
}
```

## 🚀 **Преимущества**

1. **🎯 Максимальная точность** - два независимых метода
2. **🔍 Научный подход** - видишь где методы согласны/расходятся
3. **🛡️ Надежность** - если один не работает, есть второй
4. **📈 Больше информации** - детали от каждого метода
5. **⚡ Быстрота** - параллельное выполнение

**Приложение работает на: http://localhost:8081** 🎉