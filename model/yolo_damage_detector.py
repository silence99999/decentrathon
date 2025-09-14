#!/usr/bin/env python3
"""
YOLOv8 интегратор для анализа повреждений автомобилей
Использует загруженные локальные модели yolov8n.pt и yolov8s.pt
"""

import sys
import json
import os
import numpy as np
import cv2
from pathlib import Path

# Проверяем доступность ultralytics
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("ultralytics не установлен, используем fallback режим", file=sys.stderr)

# Пути к моделям
MODELS_DIR = Path("models_download")
YOLO_MODELS = {
    "nano": MODELS_DIR / "yolov8n.pt",
    "small": MODELS_DIR / "yolov8s.pt",
    "medium": MODELS_DIR / "yolov8m.pt"
}

# Классы повреждений (COCO classes + custom damage classes)
DAMAGE_CLASSES = {
    # Стандартные COCO классы автомобилей
    "car": 2,
    "truck": 7,
    "bus": 5,
    # Кастомные классы повреждений (если модель специализированная)
    "scratch": 80,
    "dent": 81,
    "rust": 82,
    "crack": 83,
    "dirt": 84
}

def load_yolo_model(model_size="small"):
    """Загружает YOLOv8 модель указанного размера."""
    if not YOLO_AVAILABLE:
        return None

    model_path = YOLO_MODELS.get(model_size)
    if not model_path or not model_path.exists():
        # Пробуем другие доступные модели
        for size, path in YOLO_MODELS.items():
            if path.exists():
                print(f"Используем {size} модель вместо {model_size}", file=sys.stderr)
                model_path = path
                break
        else:
            print("Ни одна YOLO модель не найдена", file=sys.stderr)
            return None

    try:
        model = YOLO(str(model_path))
        print(f"Загружена YOLO модель: {model_path.name}", file=sys.stderr)
        return model
    except Exception as e:
        print(f"Ошибка загрузки YOLO модели: {e}", file=sys.stderr)
        return None

def detect_with_yolo(model, image_path, confidence=0.5):  # Повышен порог уверенности
    """Детекция объектов с помощью YOLOv8."""
    try:
        results = model(image_path, conf=confidence, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Получаем координаты и класс
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())

                    x1, y1, x2, y2 = xyxy
                    width = x2 - x1
                    height = y2 - y1
                    center_x = x1 + width / 2
                    center_y = y1 + height / 2

                    # Определяем тип объекта
                    class_name = model.names[cls] if cls < len(model.names) else f"class_{cls}"

                    detection = {
                        "class": class_name,
                        "class_id": cls,
                        "confidence": float(conf),
                        "x": float(center_x),
                        "y": float(center_y),
                        "width": float(width),
                        "height": float(height),
                        "area": float(width * height)
                    }

                    detections.append(detection)

        return detections
    except Exception as e:
        print(f"Ошибка YOLO детекции: {e}", file=sys.stderr)
        return []

def analyze_damage_from_detections(detections, image_shape):
    """Анализирует повреждения на основе детекций YOLO."""
    h, w = image_shape[:2]
    total_area = h * w

    # Счетчики повреждений
    damage_counts = {
        "scratches": 0,
        "dents": 0,
        "rust": 0,
        "dirt": 0,
        "cracks": 0
    }

    confidence_scores = {
        "scratches": 0.0,
        "dents": 0.0,
        "rust": 0.0,
        "dirt": 0.0,
        "cracks": 0.0
    }

    damage_details = []
    car_detected = False

    for detection in detections:
        class_name = detection["class"].lower()
        confidence = detection["confidence"]

        # Проверяем, есть ли автомобиль в кадре
        if class_name in ["car", "truck", "bus", "vehicle"]:
            car_detected = True

        # Классифицируем как повреждения
        damage_type = None
        if "scratch" in class_name or "scrape" in class_name:
            damage_type = "scratches"
        elif "dent" in class_name or "damage" in class_name:
            damage_type = "dents"
        elif "rust" in class_name or "corrosion" in class_name:
            damage_type = "rust"
        elif "dirt" in class_name or "mud" in class_name or "stain" in class_name:
            damage_type = "dirt"
        elif "crack" in class_name or "break" in class_name:
            damage_type = "cracks"

        if damage_type:
            damage_counts[damage_type] += 1
            confidence_scores[damage_type] = max(confidence_scores[damage_type], confidence)

            # Создаем детальную информацию
            severity = "severe" if confidence > 0.8 else "moderate" if confidence > 0.5 else "minor"

            damage_detail = {
                "type": damage_type.rstrip('s'),  # Убираем 's' в конце
                "confidence": confidence,
                "x": detection["x"],
                "y": detection["y"],
                "width": detection["width"],
                "height": detection["height"],
                "area": detection["area"],
                "severity": severity,
                "description": f"{damage_type.rstrip('s').title()} обнаружен YOLO в позиции ({int(detection['x'])}, {int(detection['y'])})",
                "detected_by": "YOLOv8 детектор"
            }
            damage_details.append(damage_detail)

    # Если автомобиль не обнаружен, применяем более агрессивную интерпретацию
    if not car_detected:
        # Считаем все неизвестные объекты потенциальными повреждениями
        for detection in detections:
            if detection["class_id"] > 80:  # Неизвестные классы
                confidence = detection["confidence"]
                if confidence > 0.4:  # Высокая уверенность
                    # Классифицируем по размеру и форме
                    area_ratio = detection["area"] / total_area
                    aspect_ratio = detection["width"] / detection["height"]

                    if area_ratio < 0.01 and aspect_ratio > 3:  # Длинный узкий объект
                        damage_counts["scratches"] += 1
                        confidence_scores["scratches"] = max(confidence_scores["scratches"], confidence * 0.8)
                    elif area_ratio < 0.02 and 0.5 < aspect_ratio < 2:  # Круглый/овальный объект
                        damage_counts["dents"] += 1
                        confidence_scores["dents"] = max(confidence_scores["dents"], confidence * 0.7)
                    elif area_ratio < 0.05:  # Средний объект
                        damage_counts["dirt"] += 1
                        confidence_scores["dirt"] = max(confidence_scores["dirt"], confidence * 0.6)

    return damage_counts, confidence_scores, damage_details

def opencv_fallback_analysis(image_path):
    """Запасной анализ через OpenCV если YOLO недоступен."""
    image = cv2.imread(image_path)
    if image is None:
        return None

    # Простой анализ на основе контуров и цветов
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Детекция краев (потенциальные повреждения)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    damage_count = len([c for c in contours if cv2.contourArea(c) > 100])

    return {
        "scratches": damage_count > 20,
        "dents": damage_count > 10,
        "rust": False,  # Сложно определить без цветового анализа
        "dirt": damage_count > 30,
        "cracks": damage_count > 15,
        "cleanliness": 1.0 - min(damage_count / 50.0, 0.8),
        "status": "Требует внимания" if damage_count > 15 else "Удовлетворительное",
        "description": f"Обнаружено {damage_count} потенциальных повреждений через OpenCV анализ",
        "detection_counts": {
            "scratches": max(0, damage_count - 15),
            "dents": max(0, damage_count - 20),
            "rust": 0,
            "dirt": max(0, damage_count - 10),
            "cracks": max(0, damage_count - 25)
        },
        "confidence_scores": {
            "scratches": 0.3 if damage_count > 20 else 0.0,
            "dents": 0.3 if damage_count > 10 else 0.0,
            "rust": 0.0,
            "dirt": 0.4 if damage_count > 30 else 0.0,
            "cracks": 0.3 if damage_count > 15 else 0.0
        },
        "damage_details": [],
        "method": "opencv_fallback"
    }

def analyze_with_yolo(image_path, model_size="small"):
    """Основная функция анализа через YOLOv8."""

    # Проверяем существование файла
    if not os.path.exists(image_path):
        return {"error": "Файл изображения не найден"}

    # Если YOLO недоступен, используем fallback
    if not YOLO_AVAILABLE:
        result = opencv_fallback_analysis(image_path)
        return result if result else {"error": "Не удалось проанализировать изображение"}

    # Загружаем YOLO модель
    model = load_yolo_model(model_size)
    if not model:
        result = opencv_fallback_analysis(image_path)
        return result if result else {"error": "Не удалось загрузить YOLO модель"}

    try:
        # Загружаем изображение для получения размеров
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Не удалось загрузить изображение"}

        # Выполняем детекцию
        detections = detect_with_yolo(model, image_path)

        # Анализируем повреждения
        damage_counts, confidence_scores, damage_details = analyze_damage_from_detections(
            detections, image.shape
        )

        # Определяем наличие повреждений (отключаем все кроме грязи)
        has_scratches = False  # Отключено для избежания ложных срабатываний
        has_dents = False      # Отключено
        has_rust = False       # Отключено
        has_dirt = confidence_scores["dirt"] > 0.4  # Оставляем только обнаружение грязи
        has_cracks = False     # Отключено

        # Вычисляем чистоту
        dirt_impact = confidence_scores["dirt"] * 0.8
        cleanliness = max(0.0, 1.0 - dirt_impact)

        # Определяем общий статус
        total_issues = sum([has_scratches, has_dents, has_rust, has_dirt, has_cracks])

        if total_issues >= 3:
            status = "Плохое"
            description = "Обнаружены множественные повреждения, требуется комплексный ремонт"
        elif total_issues >= 2:
            status = "Требует внимания"
            description = "Найдены серьезные повреждения, рекомендуется ремонт"
        elif total_issues >= 1:
            status = "Удовлетворительное"
            description = "Обнаружены незначительные повреждения"
        else:
            status = "Хорошее"
            description = "Серьезных повреждений не обнаружено"

        result = {
            "scratches": has_scratches,
            "dents": has_dents,
            "rust": has_rust,
            "dirt": has_dirt,
            "cracks": has_cracks,
            "cleanliness": cleanliness,
            "status": status,
            "description": description,
            "detection_counts": damage_counts,
            "confidence_scores": confidence_scores,
            "damage_details": damage_details,
            "method": f"yolo_{model_size} ({len(detections)} детекций)"
        }

        return result

    except Exception as e:
        print(f"Ошибка анализа: {e}", file=sys.stderr)
        # Возвращаем fallback результат
        result = opencv_fallback_analysis(image_path)
        return result if result else {"error": str(e)}

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Требуется путь к изображению"}))
        sys.exit(1)

    image_path = sys.argv[1]
    model_size = sys.argv[2] if len(sys.argv) > 2 else "small"

    try:
        result = analyze_with_yolo(image_path, model_size)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()