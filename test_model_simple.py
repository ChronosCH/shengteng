#!/usr/bin/env python3
"""
Simple test to verify TFNet model functionality
"""

import sys
import os
import numpy as np
import mindspore as ms
from mindspore import Tensor

# Add training directory to path
sys.path.append('training')

from tfnet_model import TFNetModel
from config_manager import ConfigManager

def test_model():
    """Test TFNet model with simple data"""
    print("=== TFNet Model Test ===")
    
    # Load config
    config_path = "training/configs/tfnet_config.json"
    config_manager = ConfigManager(config_path)
    config = config_manager.config
    
    print(f"Config loaded: {config['model']['name']}")
    
    # Set MindSpore context
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
    
    # Create model
    vocab_size = 100  # Small vocab for testing
    hidden_size = 512
    model = TFNetModel(hidden_size=hidden_size, word_set_num=vocab_size)
    print("✓ Model created successfully")
    
    # Create test data
    batch_size = 1
    seq_length = 50  # Shorter sequence for testing
    height = 224
    width = 224
    channels = 3
    
    # Create dummy video data
    video_data = np.random.randn(batch_size, seq_length, channels, height, width).astype(np.float32)
    video_lengths = [seq_length]  # Original lengths
    
    print(f"Test data shape: {video_data.shape}")
    print(f"Video lengths: {video_lengths}")
    
    # Convert to tensors
    video_tensor = Tensor(video_data, ms.float32)
    length_tensor = Tensor(video_lengths, ms.int32)
    
    print("✓ Test data created")
    
    try:
        # Forward pass
        print("Running forward pass...")
        output = model(video_tensor, length_tensor, is_train=True)
        print(f"✓ Forward pass successful!")
        print(f"Output shape: {output.shape}")
        print(f"Output type: {type(output)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model()
    if success:
        print("\n🎉 Model test passed!")
    else:
        print("\n❌ Model test failed!")
