# Roboflow Integration Setup

This application now integrates with multiple Roboflow models for comprehensive car damage and cleanliness detection.

## Models Used

1. **Car Scratch and Dent Detection** (car-scratch-and-dent/3)
   - Detects scratches and dents on vehicles

2. **Rust and Scratch Detection** (rust-and-scrach/1)
   - Specialized in detecting rust and scratches

3. **Car Scratch Detection** (car-scratch-xgxzs/1)
   - Additional scratch detection model for improved accuracy

4. **Dirt Detection** (dirt-fges7/9)
   - Detects dirt and cleanliness issues

## Setup Instructions

### 1. Get Your Roboflow API Key

1. Sign up or log in at [Roboflow](https://app.roboflow.com)
2. Navigate to Settings → API Keys
3. Copy your API key

### 2. Configure Environment

Create a `.env` file in the project root:

```bash
ROBOFLOW_API_KEY=your_api_key_here
```

Or set it as an environment variable:

```bash
# Windows
set ROBOFLOW_API_KEY=your_api_key_here

# Linux/Mac
export ROBOFLOW_API_KEY=your_api_key_here
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
go run .
```

The application will now use the Roboflow models for analysis. If the API key is not set or the models fail, it will automatically fallback to the ONNX model or mock analyzer.

## Detection Capabilities

The integrated models can detect:
- Scratches
- Dents
- Rust
- Cracks
- Dirt

Each detection includes:
- Confidence scores
- Detection counts
- Overall cleanliness score
- Status assessment (Good, Fair, Needs Attention, Poor)

## API Response

The enhanced response now includes:

```json
{
  "has_rust": false,
  "has_cracks": false,
  "has_dirt": false,
  "has_scratches": true,
  "has_dents": false,
  "cleanliness_score": 0.85,
  "overall_status": "Needs Attention",
  "details": "Issues found: scratches (3 areas)",
  "detection_counts": {
    "rust": 0,
    "cracks": 0,
    "scratches": 3,
    "dents": 0,
    "dirt": 0
  },
  "confidence_scores": {
    "rust": 0.0,
    "cracks": 0.0,
    "scratches": 0.75,
    "dents": 0.0,
    "dirt": 0.15
  }
}
```