#!/usr/bin/env python3
"""
简化的GPU训练脚本 - 专门解决cuBLAS问题
"""

import os
import sys
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context, Tensor

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def safe_gpu_setup():
    """安全的GPU设置，如果失败则回退到CPU"""
    try:
        print("尝试设置GPU上下文...")
        context.set_context(
            mode=context.PYNATIVE_MODE,
            device_target="GPU",
            device_id=0,
            # 关键：设置更保守的内存管理
            max_device_memory="4GB"
        )
        
        # 测试基本的GPU操作
        test_a = Tensor(np.random.randn(2, 3).astype(np.float32))
        test_b = Tensor(np.random.randn(3, 4).astype(np.float32))
        result = ops.matmul(test_a, test_b)
        
        print(f"✅ GPU设置成功，测试矩阵乘法结果形状: {result.shape}")
        return "GPU"
        
    except Exception as e:
        print(f"❌ GPU设置失败: {e}")
        print("切换到CPU模式...")
        context.set_context(
            mode=context.PYNATIVE_MODE,
            device_target="CPU"
        )
        print("✅ CPU模式设置成功")
        return "CPU"

def create_simple_model(vocab_size, device_target):
    """创建简化的模型来避免复杂的维度问题"""
    class SimpleTFNet(nn.Cell):
        def __init__(self, vocab_size):
            super(SimpleTFNet, self).__init__()
            # 使用更简单的架构
            self.conv_features = nn.SequentialCell([
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, pad_mode='pad'),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4))  # 固定输出大小
            ])
            
            self.temporal_features = nn.SequentialCell([
                nn.Conv1d(32 * 16, 64, kernel_size=3, padding=1, pad_mode='pad'),
                nn.ReLU()
            ])
            
            # 更简单的分类器
            self.classifier = nn.Dense(64, vocab_size)
            
        def construct(self, x):
            # x shape: (B, T, C, H, W)
            B, T, C, H, W = x.shape
            
            # 处理每一帧
            x = x.view(B * T, C, H, W)
            features = self.conv_features(x)  # (B*T, 32, 4, 4)
            features = features.view(B * T, -1)  # (B*T, 32*16)
            
            # 重塑为时序数据
            features = features.view(B, T, -1)  # (B, T, 32*16)
            features = features.transpose(0, 2, 1)  # (B, 32*16, T)
            
            # 时序处理
            temporal_out = self.temporal_features(features)  # (B, 64, T)
            temporal_out = temporal_out.transpose(0, 2, 1)  # (B, T, 64)
            
            # 分类
            output = self.classifier(temporal_out)  # (B, T, vocab_size)
            
            return output
    
    return SimpleTFNet(vocab_size)

def main():
    print("开始简化的GPU训练测试...")
    
    # 安全设置GPU
    device = safe_gpu_setup()
    
    # 模拟数据
    batch_size = 2
    seq_len = 8
    vocab_size = 100
    
    print(f"创建测试数据 (device: {device})...")
    video_data = Tensor(np.random.randn(batch_size, seq_len, 3, 64, 64).astype(np.float32))
    labels = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_len)).astype(np.int32))
    
    print(f"输入数据形状: {video_data.shape}")
    
    # 创建简化模型
    model = create_simple_model(vocab_size, device)
    
    # 创建损失函数和优化器
    loss_fn = nn.CrossEntropyLoss()
    optimizer = nn.Adam(model.trainable_params(), learning_rate=0.001)
    
    # 定义训练步骤
    def forward_fn(data, target):
        logits = model(data)
        # 重塑logits和target以匹配CrossEntropyLoss的要求
        logits = logits.view(-1, vocab_size)  # (B*T, vocab_size)
        target = target.view(-1)  # (B*T,)
        loss = loss_fn(logits, target)
        return loss
    
    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)
    
    def train_step(data, target):
        loss, grads = grad_fn(data, target)
        optimizer(grads)
        return loss
    
    # 训练几个步骤
    print("开始训练步骤...")
    for step in range(5):
        try:
            loss = train_step(video_data, labels)
            print(f"  步骤 {step+1}/5 - Loss: {loss.asnumpy():.4f}")
        except Exception as e:
            print(f"  ❌ 步骤 {step+1} 失败: {e}")
            if "cuBLAS" in str(e):
                print("  检测到cuBLAS错误 - 这是GPU/CUDA版本兼容性问题")
                break
            continue
    
    print("✅ 简化训练测试完成!")

if __name__ == "__main__":
    main()
