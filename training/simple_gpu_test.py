#!/usr/bin/env python3
"""
简单的GPU测试脚本
"""

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context, Tensor
import numpy as np

print("设置GPU上下文...")
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU", device_id=0)

print("创建简单的测试张量...")
# 创建一些测试张量
a = Tensor(np.random.randn(10, 20).astype(np.float32))
b = Tensor(np.random.randn(20, 15).astype(np.float32))

print(f"张量a形状: {a.shape}")
print(f"张量b形状: {b.shape}")

print("测试基本矩阵乘法...")
try:
    matmul = ops.MatMul()
    result = matmul(a, b)
    print(f"✅ 基本矩阵乘法成功! 结果形状: {result.shape}")
except Exception as e:
    print(f"❌ 基本矩阵乘法失败: {e}")

print("测试简单的线性层...")
try:
    linear = nn.Dense(20, 15)
    linear_result = linear(a)
    print(f"✅ 线性层测试成功! 结果形状: {linear_result.shape}")
except Exception as e:
    print(f"❌ 线性层测试失败: {e}")

print("测试LSTM...")
try:
    lstm = nn.LSTM(input_size=15, hidden_size=10, num_layers=1, batch_first=False)
    # 输入形状 (seq_len, batch, input_size)
    test_input = Tensor(np.random.randn(5, 2, 15).astype(np.float32))
    output, _ = lstm(test_input)
    print(f"✅ LSTM测试成功! 输出形状: {output.shape}")
except Exception as e:
    print(f"❌ LSTM测试失败: {e}")

print("所有基本测试完成!")
