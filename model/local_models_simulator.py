#!/usr/bin/env python3
"""
Локальная замена 4 моделей Roboflow с использованием существующей ONNX модели и OpenCV.
Полностью автономная работа без интернета и API ключей.
"""

import sys
import json
import os
import base64
import numpy as np
import cv2
from PIL import Image
import io
import onnxruntime as ort

# Загружаем существующую ONNX модель
ONNX_MODEL_PATH = "model/car_condition_model_compressed.onnx"

def load_onnx_model():
    """Загружает ONNX модель если доступна."""
    try:
        if os.path.exists(ONNX_MODEL_PATH):
            return ort.InferenceSession(ONNX_MODEL_PATH)
        return None
    except:
        return None

def encode_image_to_base64(image_path):
    """Кодирует изображение в base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def detect_scratches_opencv(image):
    """Обнаружение царапин с помощью OpenCV."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Применяем фильтр Собеля для обнаружения линий
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)

    # Пороговая обработка
    _, thresh = cv2.threshold(sobel.astype(np.uint8), 30, 255, cv2.THRESH_BINARY)

    # Морфологические операции
    kernel = np.ones((3, 1), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Поиск контуров
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    scratches = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        # Фильтруем по размеру и соотношению сторон
        if area > 100 and (w > h * 2 or h > w * 2):
            scratches.append({
                "x": float(x + w/2),
                "y": float(y + h/2),
                "width": float(w),
                "height": float(h),
                "confidence": min(0.3 + (area / 10000), 0.9),
                "class": "scratch"
            })

    return scratches[:5]  # Максимум 5 царапин

def detect_dents_opencv(image):
    """Обнаружение вмятин с помощью OpenCV."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Размытие для выделения крупных областей
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # Поиск темных областей (вмятины дают тени)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = 255 - thresh  # Инвертируем

    # Морфологические операции
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Поиск контуров
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dents = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        # Фильтруем по размеру и форме
        if 1000 < area < 50000:
            aspect_ratio = w / h
            if 0.5 < aspect_ratio < 2.0:  # Примерно круглые/овальные
                dents.append({
                    "x": float(x + w/2),
                    "y": float(y + h/2),
                    "width": float(w),
                    "height": float(h),
                    "confidence": min(0.3 + (area / 20000), 0.85),
                    "class": "dent"
                })

    return dents[:3]  # Максимум 3 вмятины

def detect_rust_opencv(image):
    """Обнаружение ржавчины с помощью OpenCV."""
    # Конвертируем в HSV для лучшего детектирования цвета
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Диапазон для коричневых/оранжевых цветов (ржавчина) - более строгий
    lower_rust1 = np.array([10, 100, 100])
    upper_rust1 = np.array([20, 255, 255])

    lower_rust2 = np.array([165, 100, 100])
    upper_rust2 = np.array([175, 255, 255])

    # Создаем маски
    mask1 = cv2.inRange(hsv, lower_rust1, upper_rust1)
    mask2 = cv2.inRange(hsv, lower_rust2, upper_rust2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Морфологические операции
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    # Поиск контуров
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rust_areas = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        if area > 2000:  # Увеличен минимальный размер для уменьшения ложных срабатываний
            rust_areas.append({
                "x": float(x + w/2),
                "y": float(y + h/2),
                "width": float(w),
                "height": float(h),
                "confidence": min(0.4 + (area / 15000), 0.8),
                "class": "rust"
            })

    return rust_areas[:4]  # Максимум 4 области ржавчины

def detect_dirt_opencv(image):
    """Обнаружение грязи с помощью OpenCV."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Поиск темных пятен
    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    # Морфологические операции
    kernel = np.ones((7, 7), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Поиск контуров
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dirt_areas = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        if 300 < area < 20000:  # Размер пятен грязи
            dirt_areas.append({
                "x": float(x + w/2),
                "y": float(y + h/2),
                "width": float(w),
                "height": float(h),
                "confidence": min(0.3 + (area / 10000), 0.7),
                "class": "dirt"
            })

    return dirt_areas[:6]  # Максимум 6 пятен грязи

def create_damage_detail(prediction, damage_type, model_name):
    """Создает детальную информацию о повреждении."""
    x = prediction.get("x", 0)
    y = prediction.get("y", 0)
    width = prediction.get("width", 0)
    height = prediction.get("height", 0)
    confidence = prediction.get("confidence", 0)

    # Вычисляем площадь
    area = width * height

    # Определяем серьезность
    if area > 10000 and confidence > 0.7:
        severity = "severe"
    elif area > 5000 and confidence > 0.5:
        severity = "moderate"
    else:
        severity = "minor"

    # Создаем описания
    descriptions = {
        "scratch": f"Царапина обнаружена в позиции ({int(x)}, {int(y)}) размером {width:.0f}x{height:.0f}px",
        "dent": f"Вмятина найдена в позиции ({int(x)}, {int(y)}) площадью {width:.0f}x{height:.0f}px",
        "rust": f"Ржавчина обнаружена в ({int(x)}, {int(y)}) на площади {width:.0f}x{height:.0f}px",
        "dirt": f"Загрязнение найдено в ({int(x)}, {int(y)}) размером {width:.0f}x{height:.0f}px"
    }

    return {
        "type": damage_type,
        "confidence": float(confidence),
        "x": float(x),
        "y": float(y),
        "width": float(width),
        "height": float(height),
        "area": float(area),
        "severity": severity,
        "description": descriptions.get(damage_type, f"{damage_type} в ({int(x)}, {int(y)})"),
        "detected_by": model_name
    }

def analyze_with_onnx(image, session):
    """Анализ с использованием ONNX модели."""
    try:
        # Подготавливаем изображение
        img_resized = cv2.resize(image, (224, 224))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        img_batch = np.expand_dims(img_transposed, axis=0)

        # Получаем входы модели
        input_name = session.get_inputs()[0].name

        # Запускаем предсказание
        outputs = session.run(None, {input_name: img_batch})

        # Интерпретируем результаты (зависит от архитектуры модели)
        predictions = outputs[0][0]

        # Возвращаем базовые метрики
        return {
            "has_damage": bool(predictions.max() > 0.5),
            "confidence": float(predictions.max()),
            "damage_score": float(predictions.mean())
        }
    except Exception as e:
        print(f"Ошибка ONNX анализа: {e}", file=sys.stderr)
        return {"has_damage": False, "confidence": 0.0, "damage_score": 0.0}

def analyze_image_local(image_path):
    """Локальный анализ изображения без API."""

    # Загружаем изображение
    image = cv2.imread(image_path)
    if image is None:
        raise Exception(f"Не удалось загрузить изображение: {image_path}")

    # Загружаем ONNX модель
    onnx_session = load_onnx_model()

    # Симулируем результаты 4 специализированных моделей

    # 1. Модель царапин и вмятин
    scratches = detect_scratches_opencv(image)
    dents = detect_dents_opencv(image)

    # 2. Модель ржавчины
    rust_areas = detect_rust_opencv(image)

    # 3. Модель грязи
    dirt_areas = detect_dirt_opencv(image)

    # 4. Основная ONNX модель (если доступна)
    onnx_result = {"has_damage": False, "confidence": 0.0}
    if onnx_session:
        onnx_result = analyze_with_onnx(image, onnx_session)

    # Объединяем все детекции
    all_detections = {
        "scratches": scratches,
        "dents": dents,
        "rust": rust_areas,
        "dirt": dirt_areas,
        "cracks": []  # Трещины пока не реализованы
    }

    # Вычисляем confidence scores
    confidence_scores = {
        "scratches": max([s.get("confidence", 0) for s in scratches] + [0]),
        "dents": max([d.get("confidence", 0) for d in dents] + [0]),
        "rust": max([r.get("confidence", 0) for r in rust_areas] + [0]),
        "dirt": max([d.get("confidence", 0) for d in dirt_areas] + [0]),
        "cracks": 0.0
    }

    # Создаем детальную информацию о повреждениях
    damage_details = []

    # Добавляем детали для каждого типа повреждений
    for scratch in scratches:
        damage_details.append(create_damage_detail(scratch, "scratch", "OpenCV Детектор царапин"))

    for dent in dents:
        damage_details.append(create_damage_detail(dent, "dent", "OpenCV Детектор вмятин"))

    for rust in rust_areas:
        damage_details.append(create_damage_detail(rust, "rust", "OpenCV Детектор ржавчины"))

    for dirt in dirt_areas:
        damage_details.append(create_damage_detail(dirt, "dirt", "OpenCV Детектор грязи"))

    # Вычисляем общую чистоту
    cleanliness_score = 1.0 - (confidence_scores["dirt"] if confidence_scores["dirt"] > 0 else 0.0)

    # Определяем наличие проблем с более высокими порогами
    has_rust = confidence_scores["rust"] > 0.6  # Повышен порог для уменьшения ложных срабатываний
    has_cracks = confidence_scores["cracks"] > 0.6
    has_dirt = confidence_scores["dirt"] > 0.4  # Оставляем ниже для грязи
    has_scratches = confidence_scores["scratches"] > 0.6
    has_dents = confidence_scores["dents"] > 0.6

    # Создаем список проблем
    issues = []
    if has_rust:
        issues.append(f"ржавчина ({len(all_detections['rust'])} областей)")
    if has_cracks:
        issues.append(f"трещины ({len(all_detections['cracks'])} областей)")
    if has_scratches:
        issues.append(f"царапины ({len(all_detections['scratches'])} областей)")
    if has_dents:
        issues.append(f"вмятины ({len(all_detections['dents'])} областей)")
    if has_dirt:
        issues.append(f"грязь ({len(all_detections['dirt'])} областей)")

    # Определяем общий статус
    if len(issues) >= 3:
        status = "Плохое"
        description = "Рекомендуется комплексный ремонт и покраска поврежденных участков"
    elif len(issues) >= 1:
        status = "Требует внимания"
        description = "Рекомендуется устранить повреждения для сохранения стоимости автомобиля"
    elif cleanliness_score < 0.7:
        status = "Удовлетворительное"
        description = "Рекомендуется профессиональная мойка и полировка"
    else:
        status = "Хорошее"
        description = "Автомобиль находится в отличном состоянии"

    # Формируем результат
    result = {
        "rust": has_rust,
        "cracks": has_cracks,
        "dirt": has_dirt,
        "scratches": has_scratches,
        "dents": has_dents,
        "cleanliness": float(cleanliness_score),
        "status": status,
        "description": description,
        "detection_counts": {
            "rust": len(all_detections["rust"]),
            "cracks": len(all_detections["cracks"]),
            "scratches": len(all_detections["scratches"]),
            "dents": len(all_detections["dents"]),
            "dirt": len(all_detections["dirt"])
        },
        "confidence_scores": confidence_scores,
        "damage_details": damage_details,
        "method": f"local_models (ONNX: {'доступна' if onnx_session else 'недоступна'})"
    }

    return result

def main():
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Требуется путь к изображению"}))
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(json.dumps({"error": "Файл изображения не найден"}))
        sys.exit(1)

    try:
        result = analyze_image_local(image_path)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()