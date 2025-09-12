#!/usr/bin/env python3
"""
GPU Training Test Script
Simple test to verify GPU training setup works correctly
"""

import os
import sys
import time
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_gpu_availability():
    """Test if GPU is available and working"""
    print("Testing GPU availability...")
    
    try:
        import mindspore as ms
        from mindspore import context, Tensor
        
        print(f"MindSpore version: {ms.__version__}")
        
        # Set GPU context
        context.set_context(device_target="GPU", device_id=0)
        print("✓ GPU context set successfully")
        
        # Create test tensors
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], ms.float32)
        b = Tensor([[2.0, 1.0], [1.0, 2.0]], ms.float32)
        
        # Test basic operations
        c = a + b
        d = a * b
        e = ms.ops.MatMul()(a, b)
        
        print("✓ Basic GPU tensor operations successful")
        
        # Test larger tensors
        large_a = Tensor(np.random.randn(1000, 1000), ms.float32)
        large_b = Tensor(np.random.randn(1000, 1000), ms.float32)
        
        start_time = time.time()
        large_c = ms.ops.MatMul()(large_a, large_b)
        gpu_time = time.time() - start_time
        
        print(f"✓ Large tensor multiplication completed in {gpu_time:.4f}s")
        
        return True
        
    except Exception as e:
        print(f"✗ GPU test failed: {e}")
        return False

def test_model_creation():
    """Test model creation on GPU"""
    print("\nTesting model creation...")
    
    try:
        from tfnet_model import TFNetModel
        
        # Create model
        model = TFNetModel(
            hidden_size=512,  # Smaller for testing
            word_set_num=100,
            device_target="GPU"
        )
        
        print("✓ TFNet model created successfully")
        
        # Test forward pass with dummy data
        import mindspore as ms
        from mindspore import Tensor
        
        # Create dummy input (batch_size=1, seq_len=10, channels=3, height=224, width=224)
        dummy_input = Tensor(np.random.randn(1, 10, 3, 224, 224), ms.float32)
        dummy_len = Tensor([10], ms.int32)
        
        # Forward pass
        start_time = time.time()
        output = model(dummy_input, dummy_len, is_train=False)
        forward_time = time.time() - start_time
        
        print(f"✓ Model forward pass completed in {forward_time:.4f}s")
        print(f"  Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False

def test_data_loading():
    """Test data loading components"""
    print("\nTesting data loading...")
    
    try:
        from data_processor import build_vocabulary
        from config_manager import ConfigManager
        
        # Test config loading
        config_manager = ConfigManager("configs/gpu_config.json")
        print("✓ GPU config loaded successfully")
        
        # Test vocabulary building (if label files exist)
        dataset_config = config_manager.get_dataset_config()
        
        if (os.path.exists(dataset_config["train_label_path"]) and 
            os.path.exists(dataset_config["valid_label_path"]) and
            os.path.exists(dataset_config["test_label_path"])):
            
            word2idx, vocab_size, idx2word = build_vocabulary(
                dataset_config["train_label_path"],
                dataset_config["valid_label_path"], 
                dataset_config["test_label_path"],
                dataset_config["name"]
            )
            
            print(f"✓ Vocabulary built successfully (size: {vocab_size})")
        else:
            print("? Dataset files not found, skipping vocabulary test")
        
        return True
        
    except Exception as e:
        print(f"✗ Data loading test failed: {e}")
        return False

def test_training_setup():
    """Test training setup"""
    print("\nTesting training setup...")
    
    try:
        from train_tfnet_gpu import GPUTFNetTrainer
        
        # Try to create trainer (this tests most components)
        trainer = GPUTFNetTrainer("configs/gpu_config.json")
        print("✓ GPU trainer initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Training setup test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("GPU TRAINING SYSTEM TEST")
    print("=" * 60)
    
    # Check environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
    print(f"Conda environment: {conda_env}")
    
    if 'mindspore-gpu' not in conda_env:
        print("⚠️  Warning: Not in mindspore-gpu environment")
    
    print()
    
    # Run tests
    tests = [
        ("GPU Availability", test_gpu_availability),
        ("Model Creation", test_model_creation),
        ("Data Loading", test_data_loading),
        ("Training Setup", test_training_setup),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))
        print()
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:<20}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! GPU training setup is ready.")
        return 0
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    exit(main())
