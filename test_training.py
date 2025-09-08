#!/usr/bin/env python3
"""
Simple test script to debug training issues
"""

import sys
import os

print("=== TFNet Training Debug Test ===")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

# Test basic imports
try:
    import mindspore as ms
    print(f"✓ MindSpore imported successfully, version: {ms.__version__}")
except ImportError as e:
    print(f"✗ Failed to import MindSpore: {e}")

try:
    import numpy as np
    print(f"✓ NumPy imported successfully, version: {np.__version__}")
except ImportError as e:
    print(f"✗ Failed to import NumPy: {e}")

try:
    import cv2
    print(f"✓ OpenCV imported successfully, version: {cv2.__version__}")
except ImportError as e:
    print(f"✗ Failed to import OpenCV: {e}")

# Test config file
config_path = "training/configs/tfnet_config.json"
if os.path.exists(config_path):
    print(f"✓ Config file exists: {config_path}")
    try:
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✓ Config file loaded successfully")
        print(f"  - Dataset: {config.get('dataset', {}).get('name', 'Unknown')}")
        print(f"  - Device: {config.get('model', {}).get('device_target', 'Unknown')}")
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
else:
    print(f"✗ Config file not found: {config_path}")

# Test training script import
try:
    sys.path.append('training')
    from train_tfnet import TFNetTrainer
    print("✓ TFNetTrainer imported successfully")
    
    # Try to create trainer instance
    try:
        trainer = TFNetTrainer(config_path)
        print("✓ TFNetTrainer instance created successfully")
    except Exception as e:
        print(f"✗ Failed to create TFNetTrainer instance: {e}")
        import traceback
        traceback.print_exc()
        
except ImportError as e:
    print(f"✗ Failed to import TFNetTrainer: {e}")
    import traceback
    traceback.print_exc()

print("=== Debug Test Complete ===")
