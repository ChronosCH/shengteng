#!/usr/bin/env python3
"""
TFNet训练系统基本功能测试
"""

import os
import sys
import traceback

def test_imports():
    """测试是否可以导入所有模块"""
    print("Testing imports...")
    
    try:
        # Test standard libraries
        import json
        import logging
        import numpy as np
        print("✓ Standard libraries OK")
        
        # Test MindSpore
        import mindspore as ms
        print("✓ MindSpore OK")
        
        # Test OpenCV
        import cv2
        print("✓ OpenCV OK")
        
        # Test our modules
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        import config_manager
        print("✓ config_manager OK")
        
        import modules
        print("✓ modules OK")
        
        import data_processor
        print("✓ data_processor OK")
        
        import decoder
        print("✓ decoder OK")
        
        import tfnet_model
        print("✓ tfnet_model OK")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False

def test_config_manager():
    """Test configuration manager"""
    print("\nTesting configuration manager...")
    
    try:
        from config_manager import ConfigManager
        
        # Test default config
        config_mgr = ConfigManager()
        print("✓ Default config loaded")
        
        # Test config access
        batch_size = config_mgr.get("training.batch_size")
        print(f"✓ Config access OK (batch_size: {batch_size})")
        
        # Test config validation
        is_valid = config_mgr.validate_config()
        print(f"✓ Config validation: {'PASS' if is_valid else 'FAIL'}")
        
        return True
        
    except Exception as e:
        print(f"✗ Config manager test failed: {e}")
        traceback.print_exc()
        return False

def test_model_creation():
    """Test model creation"""
    print("\nTesting model creation...")
    
    try:
        import mindspore as ms
        from mindspore import context
        from tfnet_model import TFNetModel
        
        # Set context for CPU
        context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
        
        # Create model
        model = TFNetModel(
            hidden_size=512,  # Smaller for testing
            word_set_num=100,
            device_target="CPU",
            dataset_name="CE-CSL"
        )
        print("✓ Model created successfully")
        
        # Test model parameters
        param_count = sum(p.size for p in model.get_parameters())
        print(f"✓ Model parameters: {param_count}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        traceback.print_exc()
        return False

def test_data_processor():
    """Test data processing functions"""
    print("\nTesting data processor...")
    
    try:
        from data_processor import preprocess_words, VideoTransform
        
        # Test word preprocessing
        words = ["测试(1)", "示例{2}", "数据[3]"]
        processed = preprocess_words(words)
        print(f"✓ Word preprocessing: {words} -> {processed}")
        
        # Test video transform
        transform = VideoTransform(is_train=False, crop_size=224)
        print("✓ Video transform created")
        
        return True
        
    except Exception as e:
        print(f"✗ Data processor test failed: {e}")
        traceback.print_exc()
        return False

def test_decoder():
    """Test decoder functionality"""
    print("\nTesting decoder...")
    
    try:
        from decoder import CTCDecoder, WERCalculator
        
        # Create dummy vocabulary
        word2idx = {"": 0, "你": 1, "好": 2, "世": 3, "界": 4}
        
        # Create decoder
        decoder = CTCDecoder(
            gloss_dict=word2idx,
            num_classes=len(word2idx),
            search_mode='max',
            blank_id=0
        )
        print("✓ CTC decoder created")
        
        # Test WER calculator
        references = ["你 好 世 界"]
        hypotheses = ["你 好 世 界"]
        wer_result = WERCalculator.calculate_wer(references, hypotheses)
        print(f"✓ WER calculation: {wer_result['wer']:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"✗ Decoder test failed: {e}")
        traceback.print_exc()
        return False

def test_file_structure():
    """Test file structure"""
    print("\nTesting file structure...")
    
    required_files = [
        "training/config_manager.py",
        "training/tfnet_model.py",
        "training/modules.py",
        "training/data_processor.py",
        "training/decoder.py",
        "training/train_tfnet.py",
        "training/evaluator.py",
        "training/configs/tfnet_config.json"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False
    else:
        print("✓ All required files present")
        return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("TFNet Training System - Basic Functionality Test")
    print("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("Configuration Manager", test_config_manager),
        ("Model Creation", test_model_creation),
        ("Data Processor", test_data_processor),
        ("Decoder", test_decoder)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:.<40} {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready for use.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
