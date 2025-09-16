#!/usr/bin/env python3
"""
测试修复后的TFNet模型，验证cuBLAS错误是否解决
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

def test_model_with_various_inputs():
    """测试模型在不同输入下的表现"""
    
    # 设置GPU上下文 - 使用PYNATIVE模式以支持异常处理
    print("Setting up GPU context...")
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU", device_id=0)
    
    # 创建模型
    print("Creating TFNet model...")
    model = TFNetModel(hidden_size=64, word_set_num=100, device_target="GPU")
    
    test_cases = [
        {
            "name": "Normal case",
            "batch_size": 2,
            "sequence_length": 10,
            "channels": 3,
            "height": 64,
            "width": 64,
            "data_len": [8, 6]
        },
        {
            "name": "Single batch",
            "batch_size": 1,
            "sequence_length": 5,
            "channels": 3,
            "height": 32,
            "width": 32,
            "data_len": [3]
        },
        {
            "name": "Batch size 4 (problematic case)",
            "batch_size": 4,
            "sequence_length": 12,
            "channels": 3,
            "height": 160,
            "width": 160,
            "data_len": [10, 8, 12, 6]
        },
        {
            "name": "Edge case - minimum dimensions",
            "batch_size": 1,
            "sequence_length": 1,
            "channels": 3,
            "height": 32,
            "width": 32,
            "data_len": [1]
        },
        {
            "name": "Large batch",
            "batch_size": 8,
            "sequence_length": 15,
            "channels": 3,
            "height": 128,
            "width": 128,
            "data_len": [12, 10, 15, 8, 14, 9, 13, 11]
        }
    ]
    
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
            for j, logits in enumerate([log_probs1, log_probs2, log_probs3, log_probs4, log_probs5]):
                logits_np = logits.asnumpy()
                if np.isnan(logits_np).any():
                    print(f"  Warning: log_probs{j+1} contains NaN values")
                if np.isinf(logits_np).any():
                    print(f"  Warning: log_probs{j+1} contains Inf values")
                    
        except Exception as e:
            print(f"✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            
    print("\n=== Matrix Multiplication Stress Test ===")
    test_matmul_edge_cases()

def test_matmul_edge_cases():
    """测试矩阵乘法的边缘情况"""
    from modules import NormLinear
    
    test_cases = [
        {"input_shape": (1, 64), "name": "Minimum batch"},
        {"input_shape": (4, 64), "name": "Problematic batch size 4"},
        {"input_shape": (8, 64), "name": "Larger batch"},
        {"input_shape": (10, 5, 64), "name": "3D input (T, B, C)"},
        {"input_shape": (15, 4, 64), "name": "3D input with batch 4"},
    ]
    
    linear_layer = NormLinear(64, 100)
    
    for test_case in test_cases:
        try:
            print(f"\nTesting {test_case['name']} - Input shape: {test_case['input_shape']}")
            
            # 创建测试输入
            input_tensor = Tensor(np.random.randn(*test_case['input_shape']).astype(np.float32))
            
            # 前向传播
            output = linear_layer(input_tensor)
            
            print(f"✓ Success! Output shape: {output.shape}")
            
            # 检查输出
            output_np = output.asnumpy()
            if np.isnan(output_np).any():
                print("  Warning: Output contains NaN values")
            if np.isinf(output_np).any():
                print("  Warning: Output contains Inf values")
                
        except Exception as e:
            print(f"✗ Failed: {e}")

if __name__ == "__main__":
    print("Testing fixed TFNet model for cuBLAS errors...")
    test_model_with_various_inputs()
    print("\nAll tests completed!")
