#!/usr/bin/env python3
"""
GPU训练调试脚本
用于检测和修复导致device-side assert的问题
"""

import os
import sys
import numpy as np
import mindspore as ms
from mindspore import context, Tensor
import mindspore.ops as ops

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_manager import ConfigManager
from tfnet_model import TFNetModel
from data_processor import build_vocabulary, create_dataset

def setup_debug_context():
    """设置调试模式的MindSpore上下文"""
    print("Setting up debug context...")
    
    # 使用PYNATIVE模式以便更好的调试
    context.set_context(
        mode=context.PYNATIVE_MODE,  # 使用动态图模式便于调试
        device_target="GPU",
        device_id=0,
        save_graphs=False,
        enable_graph_kernel=False,  # 禁用图内核优化以避免问题
        max_device_memory="2GB"  # 限制内存使用
    )
    print("✓ Debug context set up successfully")

def test_simple_gpu_operation():
    """测试简单的GPU操作"""
    print("\nTesting simple GPU operations...")
    
    try:
        # 简单张量操作
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], ms.float32)
        b = Tensor([[1.0, 1.0], [1.0, 1.0]], ms.float32)
        c = a + b
        print(f"✓ Simple tensor operation successful: {c.shape}")
        
        # 索引操作
        d = a[0:1]
        print(f"✓ Indexing operation successful: {d.shape}")
        
        # 堆叠操作
        e = ops.stack([a, b], axis=0)
        print(f"✓ Stacking operation successful: {e.shape}")
        
        return True
    except Exception as ex:
        print(f"✗ Simple GPU operation failed: {ex}")
        return False

def test_model_creation():
    """测试模型创建"""
    print("\nTesting model creation...")
    
    try:
        model = TFNetModel(
            hidden_size=128,  # 减小隐藏层大小
            word_set_num=100,  # 减小词汇表大小
            device_target="GPU"
        )
        print("✓ Model created successfully")
        return model
    except Exception as ex:
        print(f"✗ Model creation failed: {ex}")
        return None

def test_model_forward():
    """测试模型前向传播"""
    print("\nTesting model forward pass...")
    
    model = test_model_creation()
    if model is None:
        return False
    
    try:
        # 创建最小的测试输入
        batch_size = 1
        seq_len = 5  # 非常短的序列
        channels = 3
        height = 64  # 小尺寸
        width = 64
        
        # 创建测试数据
        test_input = Tensor(np.random.randn(batch_size, seq_len, channels, height, width).astype(np.float32))
        test_len = Tensor([seq_len], ms.int32)
        
        print(f"Input shape: {test_input.shape}")
        print(f"Length: {test_len}")
        
        # 前向传播
        outputs = model(test_input, test_len, is_train=True)
        print(f"✓ Forward pass successful")
        print(f"Number of outputs: {len(outputs)}")
        for i, output in enumerate(outputs):
            if output is not None:
                print(f"  Output {i} shape: {output.shape}")
        
        return True
    except Exception as ex:
        print(f"✗ Forward pass failed: {ex}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loading():
    """测试数据加载"""
    print("\nTesting data loading...")
    
    try:
        config_manager = ConfigManager("configs/gpu_config.json")
        
        # 构建词汇表
        dataset_config = config_manager.get_dataset_config()
        word2idx, _, _ = build_vocabulary(
            dataset_config["train_label_path"],
            dataset_config["valid_label_path"], 
            dataset_config["test_label_path"],
            dataset_config["name"]
        )
        print(f"✓ Vocabulary loaded: {len(word2idx)} words")
        
        # 创建数据集
        dataset_config = config_manager.get_dataset_config()
        train_dataset = create_dataset(
            data_path=dataset_config["train_data_path"],
            label_path=dataset_config["train_label_path"],
            word2idx=word2idx,
            batch_size=1,
            num_workers=1,
            is_train=True,
            dataset_name=dataset_config["name"],
            prefetch_size=1,
            max_rowsize=4
        )
        
        print("✓ Dataset created successfully")
        
        # 尝试获取一个批次
        data_iter = train_dataset.create_dict_iterator()
        try:
            batch = next(data_iter)
            print("✓ Successfully loaded one batch")
            
            # 检查批次数据
            for key, value in batch.items():
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                
                # 检查数据范围
                if key in ['video', 'videoLength', 'labelLength']:
                    if hasattr(value.asnumpy(), 'min'):
                        min_val = value.asnumpy().min()
                        max_val = value.asnumpy().max()
                        print(f"    Range: [{min_val}, {max_val}]")
                        
                        # 检查是否有异常值
                        if key == 'videoLength' or key == 'labelLength':
                            if min_val < 0:
                                print(f"    ⚠️  Warning: {key} has negative values!")
                            if key == 'videoLength' and max_val > 200:
                                print(f"    ⚠️  Warning: {key} has very large values!")
            
            return True
        except Exception as ex:
            print(f"✗ Failed to load batch: {ex}")
            return False
            
    except Exception as ex:
        print(f"✗ Data loading test failed: {ex}")
        import traceback
        traceback.print_exc()
        return False

def test_full_training_step():
    """测试完整的训练步骤"""
    print("\nTesting full training step...")
    
    try:
        config_manager = ConfigManager("configs/gpu_config.json")
        
        # 测试数据加载
        if not test_data_loading():
            return False
        
        # 创建模型
        dataset_config = config_manager.get_dataset_config()
        word2idx, _, _ = build_vocabulary(
            dataset_config["train_label_path"],
            dataset_config["valid_label_path"], 
            dataset_config["test_label_path"],
            dataset_config["name"]
        )
        
        model = TFNetModel(
            hidden_size=128,  # 小模型
            word_set_num=len(word2idx),
            device_target="GPU"
        )
        
        # 创建数据集
        dataset_config = config_manager.get_dataset_config()
        train_dataset = create_dataset(
            data_path=dataset_config["train_data_path"],
            label_path=dataset_config["train_label_path"],
            word2idx=word2idx,
            batch_size=1,
            num_workers=1,
            is_train=True,
            dataset_name=dataset_config["name"],
            prefetch_size=1,
            max_rowsize=4
        )
        
        # 尝试一个训练步骤
        data_iter = train_dataset.create_dict_iterator()
        batch = next(data_iter)
        
        video = batch['video']
        video_length = batch['videoLength']
        
        print(f"Running forward pass with batch:")
        print(f"  Video shape: {video.shape}")
        print(f"  Video length: {video_length}")
        
        # 前向传播
        outputs = model(video, video_length, is_train=True)
        print("✓ Full training step successful!")
        
        return True
        
    except Exception as ex:
        print(f"✗ Full training step failed: {ex}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主调试函数"""
    print("=" * 60)
    print("GPU Training Debug Script")
    print("=" * 60)
    
    # 设置调试上下文
    setup_debug_context()
    
    # 运行测试
    tests = [
        ("Simple GPU Operations", test_simple_gpu_operation),
        ("Model Creation", lambda: test_model_creation() is not None),
        ("Model Forward Pass", test_model_forward),
        ("Data Loading", test_data_loading),
        ("Full Training Step", test_full_training_step)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print(f"{'='*60}")
        
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as ex:
            results.append((test_name, False))
            print(f"❌ {test_name}: FAILED with exception: {ex}")
    
    # 总结
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! GPU training should work.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
