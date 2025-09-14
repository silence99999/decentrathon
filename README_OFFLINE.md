# Offline Car Damage Detection

This application now includes a fully offline car damage and cleanliness detection system that doesn't require any API keys or internet connection.

## Features

The offline analyzer can detect:

- **Rust**: Using HSV color analysis to detect rust-colored areas
- **Scratches**: Edge detection and line analysis to find scratch-like features
- **Dents**: Circular/elliptical shape detection and shadow analysis
- **Dirt**: Brightness and uniformity analysis for cleanliness assessment
- **Cracks**: Morphological operations to detect crack-like structures

## Setup Instructions

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- OpenCV for computer vision operations
- NumPy for numerical computations
- Pillow for image processing
- Additional dependencies for fallback models

### 2. Optional: Setup Model Cache

If you have a Roboflow API key and want to cache model information:

```bash
cd model
python download_models.py
```

This creates a local cache structure, but the offline analyzer works without it.

### 3. Run the Application

```bash
go run .
```

The application will automatically use the offline analyzer as the primary detection method.

## How It Works

### Detection Methods

1. **Rust Detection**: Analyzes HSV color space for rust-typical orange-brown and red-brown colors
2. **Scratch Detection**: Uses Canny edge detection and Hough line transforms to find linear features
3. **Dent Detection**: Combines circular Hough transforms with dark region analysis
4. **Dirt Analysis**: Evaluates brightness distribution and color uniformity
5. **Crack Detection**: Applies morphological operations (top-hat, black-hat) to enhance thin structures

### Fallback Chain

The application uses this priority order:

1. **Offline Analyzer** (No API required) ← **PRIMARY**
2. **Roboflow API** (If `ROBOFLOW_API_KEY` is set)
3. **ONNX Model** (If model file exists)
4. **Mock Analyzer** (Always available)

## Sample Response

```json
{
  "rust": false,
  "cracks": false,
  "dirt": true,
  "scratches": true,
  "dents": false,
  "cleanliness": 0.45,
  "status": "Needs Attention",
  "description": "Issues found: dirt (2 areas), scratches (3 areas)",
  "detection_counts": {
    "rust": 0,
    "cracks": 0,
    "scratches": 3,
    "dents": 0,
    "dirt": 2
  },
  "confidence_scores": {
    "rust": 0.0,
    "cracks": 0.0,
    "scratches": 0.6,
    "dents": 0.0,
    "dirt": 0.55
  }
}
```

## Advantages

- **No API Keys Required**: Runs completely offline
- **Fast Processing**: Uses optimized OpenCV operations
- **Reliable**: Always available as primary method
- **Comprehensive**: Detects 5 different types of issues
- **Detailed Results**: Provides confidence scores and area counts

## Limitations

- **Heuristic-Based**: Uses computer vision techniques rather than trained models
- **Lighting Dependent**: Performance varies with image quality and lighting
- **Parameter Tuning**: Thresholds may need adjustment for specific use cases

The offline system provides a robust baseline for car surface analysis without external dependencies.