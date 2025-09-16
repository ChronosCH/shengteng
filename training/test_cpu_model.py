#!/usr/bin/env python3
"""
CPU模式测试脚本
"""

import os
import sys
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context, Tensor
import numpy as np

print("设置CPU上下文...")
context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tfnet_model import TFNetModel
from data_processor import build_vocabulary

def test_cpu_model():
    """在CPU上测试模型"""
    print("开始CPU模式测试...")
    
    # 构建词汇表
    train_label_path = "/data/shengteng/training/data/CE-CSL/label/train.csv"
    valid_label_path = "/data/shengteng/training/data/CE-CSL/label/dev.csv"
    test_label_path = "/data/shengteng/training/data/CE-CSL/label/test.csv"
    
    word2idx, vocab_size, idx2word = build_vocabulary(train_label_path, valid_label_path, test_label_path, "CE-CSL")
    print(f"词汇表大小: {vocab_size}")
    
    # 创建模型
    model = TFNetModel(hidden_size=128, word_set_num=vocab_size+1, device_target="CPU", dataset_name="CE-CSL")
    print("模型创建成功")
    
    # 创建模拟数据
    batch_size = 1
    seq_len = 10
    height, width = 160, 160
    channels = 3
    
    # 模拟视频数据 (B, T, C, H, W)
    video_data = Tensor(np.random.randn(batch_size, seq_len, channels, height, width).astype(np.float32))
    video_len = [seq_len]
    
    print(f"输入数据形状: {video_data.shape}")
    print(f"视频长度: {video_len}")
    
    try:
        print("开始前向传播...")
        outputs = model(video_data, video_len, is_train=True)
        print("✅ CPU模型前向传播成功!")
        
        # 检查输出
        for i, output in enumerate(outputs[:-3]):  # 排除最后的None值
            if output is not None:
                print(f"  输出 {i} 形状: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ CPU模型前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_cpu_model()
    if success:
        print("\n✅ CPU测试成功! 问题可能是GPU/CUDA相关的。")
    else:
        print("\n❌ CPU测试也失败，可能是模型结构问题。")
