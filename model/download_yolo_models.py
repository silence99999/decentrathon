#!/usr/bin/env python3
"""
Скрипт для скачивания лучших YOLOv8 моделей для анализа повреждений автомобилей
"""

import os
import requests
import sys
from pathlib import Path
import zipfile
from tqdm import tqdm

# Создаем папку для моделей
models_dir = Path("models_download")
models_dir.mkdir(exist_ok=True)

# Лучшие модели для скачивания
MODELS = {
    "yolov8n": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
        "size": "6.2MB",
        "description": "YOLOv8 Nano - самая быстрая модель"
    },
    "yolov8s": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt",
        "size": "21.5MB",
        "description": "YOLOv8 Small - баланс скорости и точности"
    },
    "yolov8m": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt",
        "size": "49.7MB",
        "description": "YOLOv8 Medium - высокая точность"
    }
}

def download_file(url, filename):
    """Скачивает файл с прогресс-баром"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(filename, 'wb') as file, tqdm(
            desc=f"Скачивание {filename.name}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    pbar.update(len(chunk))

        print(f"✅ Успешно скачано: {filename}")
        return True

    except Exception as e:
        print(f"❌ Ошибка при скачивании {filename}: {e}")
        return False

def download_huggingface_model():
    """Скачивает модель с Hugging Face"""
    try:
        # Скачиваем специализированную модель для повреждений автомобилей
        hf_url = "https://huggingface.co/nezahatkorkmaz/car-damage-level-detection-yolov8/resolve/main/best.pt"
        filename = models_dir / "car_damage_yolov8.pt"

        if download_file(hf_url, filename):
            print(f"✅ Скачана специализированная модель для повреждений автомобилей")
            return True
    except:
        print("❌ Не удалось скачать модель с Hugging Face")
        return False

    return False

def main():
    print("🔥 Скачиваем лучшие YOLOv8 модели для анализа повреждений автомобилей 🔥")
    print("=" * 80)

    success_count = 0

    # Скачиваем базовые YOLOv8 модели
    for model_name, model_info in MODELS.items():
        print(f"\n🚀 {model_info['description']} ({model_info['size']})")
        filename = models_dir / f"{model_name}.pt"

        if filename.exists():
            print(f"⏭️ Модель {model_name} уже существует, пропускаем")
            success_count += 1
            continue

        if download_file(model_info["url"], filename):
            success_count += 1

    # Скачиваем специализированную модель с Hugging Face
    print(f"\n🎯 Специализированная модель для повреждений автомобилей")
    if download_huggingface_model():
        success_count += 1

    print("\n" + "=" * 80)
    print(f"🎉 Результат: {success_count} моделей скачано успешно!")

    if success_count > 0:
        print("\n📁 Скачанные модели в папке models_download/:")
        for file in models_dir.glob("*.pt"):
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"   • {file.name} ({size_mb:.1f} MB)")

    print("\n💡 Использование:")
    print("   1. YOLOv8n - для быстрого анализа в реальном времени")
    print("   2. YOLOv8s - для общего использования (рекомендуется)")
    print("   3. YOLOv8m - для максимальной точности")
    print("   4. car_damage_yolov8.pt - специально обученная для повреждений")

if __name__ == "__main__":
    main()