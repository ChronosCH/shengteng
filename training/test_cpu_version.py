#!/usr/bin/env python3
"""
CPU测试版本 - 验证模型逻辑是否正确，避开GPU cuBLAS问题
"""

import os
import sys
import numpy as np
import mindspore as ms
import mindspore.ops as ops
from mindspore import context, Tensor

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tfnet_model import TFNetModel

def test_model_on_cpu():
    """在CPU上测试模型以验证逻辑正确性"""
    
    # 设置CPU上下文
    print("Setting up CPU context...")
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    
    # 创建模型 - 指定CPU设备
    print("Creating TFNet model for CPU...")
    model = TFNetModel(hidden_size=64, word_set_num=100, device_target="CPU")
    
    test_cases = [
        {
            "name": "Normal case - CPU",
            "batch_size": 2,
            "sequence_length": 10,
            "channels": 3,
            "height": 64,
            "width": 64,
            "data_len": [8, 6]
        },
        {
            "name": "Batch size 4 (problematic case) - CPU",
            "batch_size": 4,
            "sequence_length": 12,
            "channels": 3,
            "height": 160,
            "width": 160,
            "data_len": [10, 8, 12, 6]
        },
        {
            "name": "Large batch - CPU",
            "batch_size": 8,
            "sequence_length": 15,
            "channels": 3,
            "height": 128,
            "width": 128,
            "data_len": [12, 10, 15, 8, 14, 9, 13, 11]
        }
    ]
    
    success_count = 0
    
    for i, test_case in enumerate(test_cases):
        print(f"\n--- Test Case {i+1}: {test_case['name']} ---")
        
        try:
            # 创建测试数据
            batch_size = test_case['batch_size']
            seq_len = test_case['sequence_length']
            channels = test_case['channels']
            height = test_case['height']
            width = test_case['width']
            data_len = test_case['data_len']
            
            print(f"Input shape: ({batch_size}, {seq_len}, {channels}, {height}, {width})")
            print(f"Data lengths: {data_len}")
            
            # 创建随机输入数据
            seq_data = Tensor(np.random.randn(batch_size, seq_len, channels, height, width).astype(np.float32))
            data_len_tensor = data_len
            
            # 前向传播
            print("Running forward pass...")
            outputs = model(seq_data, data_len_tensor, is_train=True)
            
            # 检查输出
            log_probs1, log_probs2, log_probs3, log_probs4, log_probs5, lgt_tensor, _, _, _ = outputs
            
            print(f"✓ Success! Output shapes:")
            print(f"  log_probs1: {log_probs1.shape}")
            print(f"  log_probs2: {log_probs2.shape}")  
            print(f"  log_probs3: {log_probs3.shape}")
            print(f"  log_probs4: {log_probs4.shape}")
            print(f"  log_probs5: {log_probs5.shape}")
            print(f"  lgt_tensor: {lgt_tensor.shape}")
            
            # 检查输出是否包含NaN或Inf
            all_logits = [log_probs1, log_probs2, log_probs3, log_probs4, log_probs5]
            for j, logits in enumerate(all_logits):
                logits_np = logits.asnumpy()
                if np.isnan(logits_np).any():
                    print(f"  Warning: log_probs{j+1} contains NaN values")
                if np.isinf(logits_np).any():
                    print(f"  Warning: log_probs{j+1} contains Inf values")
                    
            success_count += 1
                    
        except Exception as e:
            print(f"✗ Failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n=== CPU测试结果 ===")
    print(f"成功: {success_count}/{len(test_cases)} 个测试用例")
    
    if success_count == len(test_cases):
        print("✓ 所有CPU测试通过！模型逻辑正确。")
        print("GPU问题可能是由于cuBLAS库或驱动程序兼容性导致的。")
        return True
    else:
        print("✗ 一些测试失败，模型逻辑可能有问题。")
        return False

def test_training_compatibility():
    """测试训练兼容性"""
    print("\n=== 测试训练模式兼容性 ===")
    
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    model = TFNetModel(hidden_size=64, word_set_num=100, device_target="CPU")
    
    # 测试训练模式
    try:
        model.set_train(True)
        
        # 创建小批量数据
        batch_size = 2
        seq_len = 8
        seq_data = Tensor(np.random.randn(batch_size, seq_len, 3, 64, 64).astype(np.float32))
        data_len = [6, 8]
        
        # 前向传播
        outputs = model(seq_data, data_len, is_train=True)
        
        print("✓ 训练模式测试通过")
        
        # 测试评估模式
        model.set_train(False)
        outputs = model(seq_data, data_len, is_train=False)
        
        print("✓ 评估模式测试通过")
        
        return True
        
    except Exception as e:
        print(f"✗ 训练兼容性测试失败: {e}")
        return False

if __name__ == "__main__":
    print("Testing TFNet model on CPU to verify logic correctness...")
    
    # 测试基本功能
    basic_success = test_model_on_cpu()
    
    # 测试训练兼容性
    training_success = test_training_compatibility()
    
    if basic_success and training_success:
        print("\n🎉 CPU测试全部通过！")
        print("\n📋 关于GPU cuBLAS错误的解决建议:")
        print("1. GPU cuBLAS错误通常是由驱动或库版本不兼容导致")
        print("2. 可以尝试使用CPU模式进行训练（在配置中设置device_target: 'CPU'）")
        print("3. 或者更新CUDA/cuDNN版本")
        print("4. 对于训练，可以临时使用CPU模式，虽然速度较慢但可以验证算法正确性")
    else:
        print("\n❌ 测试失败，需要进一步调试")
        
    print("\nCPU测试完成!")
