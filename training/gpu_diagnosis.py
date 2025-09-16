#!/usr/bin/env python3
"""
详细GPU问题诊断脚本
"""

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context, Tensor, Parameter
import numpy as np

print("设置GPU上下文...")
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU", device_id=0)

print("测试不同类型的矩阵乘法...")

# 测试1: 基本ops.MatMul
print("\n1. 测试基本ops.MatMul...")
a = Tensor(np.random.randn(10, 20).astype(np.float32))
b = Tensor(np.random.randn(20, 15).astype(np.float32))
try:
    matmul = ops.MatMul()
    result = matmul(a, b)
    print(f"✅ 基本MatMul成功! 结果形状: {result.shape}")
except Exception as e:
    print(f"❌ 基本MatMul失败: {e}")

# 测试2: 使用Parameter的矩阵乘法
print("\n2. 测试使用Parameter的矩阵乘法...")
try:
    weight = Parameter(Tensor(np.random.randn(20, 15).astype(np.float32)), name='weight')
    print(f"Parameter创建成功，形状: {weight.shape}")
    result2 = matmul(a, weight)
    print(f"✅ Parameter MatMul成功! 结果形状: {result2.shape}")
except Exception as e:
    print(f"❌ Parameter MatMul失败: {e}")

# 测试3: 检查nn.Dense内部
print("\n3. 分析nn.Dense层...")
try:
    linear = nn.Dense(20, 15)
    print(f"Dense层创建成功")
    print(f"权重形状: {linear.weight.shape}")
    print(f"偏置形状: {linear.bias.shape}")
    
    # 手动执行Dense层的操作
    print("手动执行Dense层操作...")
    manual_result = ops.matmul(a, linear.weight.T) + linear.bias
    print(f"✅ 手动Dense操作成功! 结果形状: {manual_result.shape}")
    
    # 尝试使用Dense层
    print("使用Dense层...")
    dense_result = linear(a)
    print(f"✅ Dense层成功! 结果形状: {dense_result.shape}")
    
except Exception as e:
    print(f"❌ Dense层测试失败: {e}")
    import traceback
    traceback.print_exc()

# 测试4: 检查数据类型和内存布局
print("\n4. 检查数据类型和内存布局...")
try:
    print(f"张量a - 数据类型: {a.dtype}, 是否连续: {a.is_contiguous()}")
    print(f"张量b - 数据类型: {b.dtype}, 是否连续: {b.is_contiguous()}")
    
    # 确保数据是连续的
    a_cont = a.contiguous()
    b_cont = b.contiguous()
    
    print("测试连续数据的矩阵乘法...")
    result3 = matmul(a_cont, b_cont)
    print(f"✅ 连续数据MatMul成功! 结果形状: {result3.shape}")
    
except Exception as e:
    print(f"❌ 连续数据测试失败: {e}")

print("\n诊断完成!")
