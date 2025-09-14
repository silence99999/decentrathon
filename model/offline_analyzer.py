#!/usr/bin/env python3
"""
Offline car damage and dirt detection using computer vision techniques.
This module provides heuristic-based analysis without requiring API calls.
"""

import sys
import json
import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import colorsys

# Configuration
MODEL_CACHE_DIR = os.path.join(os.path.dirname(__file__), "model_cache")

def load_offline_config():
    """Load offline configuration if available."""
    config_file = os.path.join(MODEL_CACHE_DIR, "offline_config.json")
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return json.load(f)
    return None

def analyze_rust_heuristic(image_array):
    """
    Detect rust-like colors and textures using heuristic methods.

    Args:
        image_array: numpy array of the image

    Returns:
        dict: rust analysis results
    """
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)

    # Define rust color ranges in HSV
    # Rust typically appears as orange-brown to red-brown colors
    rust_ranges = [
        # Orange-brown range
        (np.array([10, 50, 50]), np.array([25, 255, 255])),
        # Red-brown range
        (np.array([0, 50, 50]), np.array([10, 255, 255])),
    ]

    rust_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in rust_ranges:
        mask = cv2.inRange(hsv, lower, upper)
        rust_mask = cv2.bitwise_or(rust_mask, mask)

    # Calculate rust percentage
    total_pixels = rust_mask.size
    rust_pixels = cv2.countNonZero(rust_mask)
    rust_percentage = rust_pixels / total_pixels

    # Find contours for rust areas
    contours, _ = cv2.findContours(rust_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rust_areas = len([c for c in contours if cv2.contourArea(c) > 100])

    return {
        "detected": rust_percentage > 0.02,  # 2% threshold
        "confidence": min(rust_percentage * 10, 1.0),  # Scale to 0-1
        "area_count": rust_areas,
        "coverage_percentage": rust_percentage * 100
    }

def analyze_scratches_heuristic(image_array):
    """
    Detect scratch-like features using edge detection and line analysis.

    Args:
        image_array: numpy array of the image

    Returns:
        dict: scratch analysis results
    """
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # Detect lines using HoughLines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=20, maxLineGap=5)

    scratch_count = 0
    if lines is not None:
        # Filter lines that look like scratches (longer, thinner features)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if length > 30:  # Minimum length for a scratch
                scratch_count += 1

    # Calculate confidence based on number and length of detected lines
    confidence = min(scratch_count / 20.0, 1.0)  # Normalize to 0-1

    return {
        "detected": scratch_count > 3,
        "confidence": confidence,
        "line_count": scratch_count,
        "area_count": max(1, scratch_count // 5)  # Group lines into areas
    }

def analyze_dents_heuristic(image_array):
    """
    Detect dent-like features using shadow and curvature analysis.

    Args:
        image_array: numpy array of the image

    Returns:
        dict: dent analysis results
    """
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    # Apply bilateral filter to reduce noise while keeping edges sharp
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    # Detect circular/elliptical shapes that might be dents
    circles = cv2.HoughCircles(
        filtered,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=30,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=100
    )

    dent_count = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        dent_count = len(circles[0])

    # Also check for irregular dark areas that might be dents
    # Apply threshold to find dark regions
    _, thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours in dark areas
    contours, _ = cv2.findContours(255 - thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    irregular_areas = len([c for c in contours if 100 < cv2.contourArea(c) < 2000])

    total_dents = dent_count + (irregular_areas // 3)  # Weight irregular areas less
    confidence = min(total_dents / 10.0, 1.0)

    return {
        "detected": total_dents > 1,
        "confidence": confidence,
        "circular_count": dent_count,
        "irregular_count": irregular_areas,
        "area_count": total_dents
    }

def analyze_dirt_heuristic(image_array):
    """
    Analyze image cleanliness using brightness, contrast, and color analysis.

    Args:
        image_array: numpy array of the image

    Returns:
        dict: dirt/cleanliness analysis results
    """
    # Convert to different color spaces for analysis
    hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)

    # Analyze brightness (L channel in LAB)
    l_channel = lab[:, :, 0]
    mean_brightness = np.mean(l_channel)
    brightness_std = np.std(l_channel)

    # Analyze color saturation (S channel in HSV)
    s_channel = hsv[:, :, 1]
    mean_saturation = np.mean(s_channel)

    # Calculate cleanliness score (0-1, higher = cleaner)
    # Clean surfaces tend to be brighter and more uniform
    brightness_score = min(mean_brightness / 200.0, 1.0)  # Normalize brightness
    uniformity_score = max(0, 1.0 - (brightness_std / 100.0))  # Lower std = more uniform

    # Combine scores
    cleanliness_score = (brightness_score * 0.6 + uniformity_score * 0.4)

    # Detect dirty areas (dark, low saturation regions)
    dirty_mask = (l_channel < mean_brightness - brightness_std) & (s_channel < 50)
    dirty_percentage = np.sum(dirty_mask) / dirty_mask.size

    # Find dirty regions
    dirty_mask_uint8 = dirty_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(dirty_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dirty_areas = len([c for c in contours if cv2.contourArea(c) > 200])

    return {
        "detected": dirty_percentage > 0.15 or cleanliness_score < 0.4,
        "confidence": 1.0 - cleanliness_score,
        "cleanliness_score": cleanliness_score,
        "dirty_percentage": dirty_percentage * 100,
        "area_count": dirty_areas,
        "brightness": mean_brightness,
        "uniformity": uniformity_score
    }

def analyze_cracks_heuristic(image_array):
    """
    Detect crack-like features using morphological operations.

    Args:
        image_array: numpy array of the image

    Returns:
        dict: crack analysis results
    """
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    # Apply morphological operations to enhance crack-like structures
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Top hat operation to enhance thin bright structures
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

    # Black hat operation to enhance thin dark structures
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Combine both
    enhanced = cv2.add(tophat, blackhat)

    # Apply threshold
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours that look like cracks (thin, elongated)
    crack_count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:  # Minimum area
            # Calculate aspect ratio
            rect = cv2.minAreaRect(contour)
            width, height = rect[1]
            if width > 0 and height > 0:
                aspect_ratio = max(width, height) / min(width, height)
                if aspect_ratio > 3:  # Elongated shape
                    crack_count += 1

    confidence = min(crack_count / 15.0, 1.0)

    return {
        "detected": crack_count > 2,
        "confidence": confidence,
        "feature_count": crack_count,
        "area_count": max(1, crack_count // 3)
    }

def analyze_image_offline(image_path):
    """
    Perform comprehensive offline analysis of car surface image.

    Args:
        image_path: Path to the image file

    Returns:
        dict: Combined analysis results
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise Exception(f"Could not load image: {image_path}")

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform all analyses
    rust_result = analyze_rust_heuristic(image_rgb)
    scratch_result = analyze_scratches_heuristic(image_rgb)
    dent_result = analyze_dents_heuristic(image_rgb)
    dirt_result = analyze_dirt_heuristic(image_rgb)
    crack_result = analyze_cracks_heuristic(image_rgb)

    # Combine results
    result = {
        "rust": rust_result["detected"],
        "cracks": crack_result["detected"],
        "dirt": dirt_result["detected"],
        "scratches": scratch_result["detected"],
        "dents": dent_result["detected"],
        "cleanliness": dirt_result["cleanliness_score"],
        "detection_counts": {
            "rust": rust_result["area_count"],
            "cracks": crack_result["area_count"],
            "scratches": scratch_result["area_count"],
            "dents": dent_result["area_count"],
            "dirt": dirt_result["area_count"]
        },
        "confidence_scores": {
            "rust": rust_result["confidence"],
            "cracks": crack_result["confidence"],
            "scratches": scratch_result["confidence"],
            "dents": dent_result["confidence"],
            "dirt": dirt_result["confidence"]
        }
    }

    # Determine overall status
    issues = []
    if result["rust"]:
        issues.append(f"rust ({rust_result['area_count']} areas)")
    if result["cracks"]:
        issues.append(f"cracks ({crack_result['area_count']} areas)")
    if result["scratches"]:
        issues.append(f"scratches ({scratch_result['area_count']} areas)")
    if result["dents"]:
        issues.append(f"dents ({dent_result['area_count']} areas)")
    if result["dirt"]:
        issues.append(f"dirt ({dirt_result['area_count']} areas)")

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
        result = analyze_image_offline(image_path)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()