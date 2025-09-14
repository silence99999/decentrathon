#!/usr/bin/env python3
"""
Python wrapper for the car condition detection model using ONNX.
"""

import sys
import json
import os
import numpy as np
import onnxruntime as ort
from PIL import Image

# Initialize ONNX model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "car_condition_model_compressed.onnx")

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess image for ONNX model input.

    Args:
        image_path: Path to the image file
        target_size: Target size for the model input

    Returns:
        numpy array: Preprocessed image
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size, Image.LANCZOS)
    img_array = np.array(img).astype('float32')
    img_array = img_array / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def analyze_image(image_path):
    """
    Analyze car condition using ONNX model.

    Args:
        image_path: Path to the image file

    Returns:
        dict: Analysis results
    """

    # Load ONNX model
    session = ort.InferenceSession(MODEL_PATH)

    # Get input name and shape
    input_name = session.get_inputs()[0].name

    # Preprocess image
    input_data = preprocess_image(image_path)

    # Run inference
    outputs = session.run(None, {input_name: input_data})

    # Model outputs:
    # Output 0 "cleanliness": 3 values (categories or scores)
    # Output 1 "damage": 4 values (different damage types)

    cleanliness_output = outputs[0][0]  # Shape: (3,)
    damage_output = outputs[1][0]  # Shape: (4,)

    # Apply softmax to get probabilities if outputs are logits
    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    cleanliness_probs = softmax(cleanliness_output)
    damage_probs = softmax(damage_output)

    # Interpret cleanliness (assuming: dirty, moderate, clean)
    cleanliness_categories = ["dirty", "moderate", "clean"]
    cleanliness_idx = np.argmax(cleanliness_probs)
    cleanliness_label = cleanliness_categories[cleanliness_idx]
    cleanliness_score = cleanliness_probs[2]  # Probability of being clean

    # Interpret damage (assuming: rust, cracks, dents, scratches or similar)
    # Using threshold of 0.3 for damage detection (adjustable)
    damage_threshold = 0.3
    damage_types = ["rust", "cracks", "dents", "scratches"]
    detected_damage = [damage_types[i] for i, prob in enumerate(damage_probs) if prob > damage_threshold]

    result = {
        "rust": bool(damage_probs[0] > damage_threshold),
        "cracks": bool(damage_probs[1] > damage_threshold),
        "dirt": bool(cleanliness_label == "dirty"),
        "cleanliness": float(cleanliness_score),
        "cleanliness_label": cleanliness_label,
        "damage_detected": detected_damage,
        "status": "Good",
        "description": "Surface appears to be in good condition"
    }

    # Determine overall status based on findings
    issues = []
    if result["rust"]:
        issues.append("rust detected")
    if result["cracks"]:
        issues.append("cracks detected")
    if result["dirt"]:
        issues.append("dirt detected")

    if len(issues) >= 3:
        result["status"] = "Плохое"
        result["description"] = "Рекомендуется комплексный ремонт и покраска поврежденных участков"
    elif len(issues) >= 1:
        result["status"] = "Требует внимания"
        result["description"] = "Рекомендуется устранить повреждения для сохранения стоимости автомобиля"
    elif result["cleanliness"] < 0.7:
        result["status"] = "Удовлетворительное"
        result["description"] = "Рекомендуется профессиональная мойка и полировка"
    else:
        result["status"] = "Хорошее"
        result["description"] = "Автомобиль находится в отличном состоянии"

    return result

def main():
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Image path required"}))
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(json.dumps({"error": "Image file not found"}))
        sys.exit(1)

    try:
        result = analyze_image(image_path)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()