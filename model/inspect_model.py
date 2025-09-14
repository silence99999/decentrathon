#!/usr/bin/env python3
"""
Inspect ONNX model structure to understand inputs and outputs.
"""

import onnxruntime as ort
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "car_condition_model_compressed.onnx")

# Load the model
session = ort.InferenceSession(MODEL_PATH)

print("=== ONNX Model Information ===\n")

# Input information
print("INPUTS:")
for i, input_info in enumerate(session.get_inputs()):
    print(f"  Input {i}:")
    print(f"    Name: {input_info.name}")
    print(f"    Shape: {input_info.shape}")
    print(f"    Type: {input_info.type}")
    print()

# Output information
print("OUTPUTS:")
for i, output_info in enumerate(session.get_outputs()):
    print(f"  Output {i}:")
    print(f"    Name: {output_info.name}")
    print(f"    Shape: {output_info.shape}")
    print(f"    Type: {output_info.type}")
    print()

# Run a test inference with dummy data to see actual output shape and values
print("=== Test Inference with Random Data ===\n")
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape

# Handle dynamic dimensions (often marked as 'N' or string)
concrete_shape = []
for dim in input_shape:
    if isinstance(dim, str) or dim == 'N':
        concrete_shape.append(1)  # Use batch size of 1
    else:
        concrete_shape.append(dim)

# Create dummy input
dummy_input = np.random.randn(*concrete_shape).astype(np.float32)
print(f"Input shape for test: {dummy_input.shape}")

# Run inference
outputs = session.run(None, {input_name: dummy_input})

print(f"\nNumber of outputs: {len(outputs)}")
for i, output in enumerate(outputs):
    print(f"\nOutput {i}:")
    print(f"  Shape: {output.shape}")
    print(f"  Data type: {output.dtype}")
    print(f"  Min value: {np.min(output)}")
    print(f"  Max value: {np.max(output)}")
    print(f"  Mean value: {np.mean(output)}")
    if output.size <= 10:
        print(f"  Values: {output.flatten()}")
    else:
        print(f"  First 10 values: {output.flatten()[:10]}")