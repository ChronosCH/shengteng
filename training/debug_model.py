#!/usr/bin/env python3
"""
模型调试脚本 - 用于诊断cuBLAS错误
"""

import os
import sys
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context, Tensor

# 设置MindSpore上下文
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU", device_id=0)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tfnet_model import TFNetModel
from data_processor import build_vocabulary, create_dataset

def debug_data_shapes():
    """调试数据形状"""
    print("开始调试数据形状...")
    
    # 构建词汇表
    train_label_path = "/data/shengteng/training/data/CE-CSL/label/train.csv"
    valid_label_path = "/data/shengteng/training/data/CE-CSL/label/dev.csv"
    test_label_path = "/data/shengteng/training/data/CE-CSL/label/test.csv"
    
    word2idx, vocab_size, idx2word = build_vocabulary(train_label_path, valid_label_path, test_label_path, "CE-CSL")
    print(f"词汇表大小: {vocab_size}")
    
    # 创建数据集
    train_data_path = "/data/shengteng/training/data/CE-CSL/video/train"
    
    dataset = create_dataset(
        data_path=train_data_path,
        label_path=train_label_path,
        word2idx=word2idx,
        dataset_name="CE-CSL",
        batch_size=1,
        is_train=True,
        num_workers=1,
        prefetch_size=1,
        max_rowsize=4,
        crop_size=160,
        max_frames=50
    )
    
    print("检查数据集中的前几个样本...")
    for i, (video, label, video_len, label_len) in enumerate(dataset):
        print(f"\n样本 {i}:")
        print(f"  视频形状: {video.shape}")
        print(f"  视频数据类型: {video.dtype}")
        print(f"  视频数据范围: [{video.min():.3f}, {video.max():.3f}]")
        print(f"  标签形状: {label.shape}")
        print(f"  视频长度: {video_len}")
        print(f"  标签长度: {label_len}")
        
        # 检查是否有异常值
        if video.shape[0] == 0 or video.shape[1] == 0:
            print(f"  ❌ 发现零维度! 视频形状: {video.shape}")
        
        if np.any(np.isnan(video.asnumpy())):
            print(f"  ❌ 发现NaN值!")
            
        if np.any(np.isinf(video.asnumpy())):
            print(f"  ❌ 发现无穷值!")
            
        if i >= 3:  # 只检查前几个样本
            break
    
    return dataset, word2idx, vocab_size

def debug_model_forward():
    """调试模型前向传播"""
    print("\n开始调试模型前向传播...")
    
    dataset, word2idx, vocab_size = debug_data_shapes()
    
    # 创建模型
    model = TFNetModel(hidden_size=128, word_set_num=vocab_size+1, device_target="GPU", dataset_name="CE-CSL")
    
    print(f"模型已创建，词汇表大小: {vocab_size+1}")
    
    # 获取一个样本
    for video, label, video_len, label_len in dataset:
        print(f"\n处理样本:")
        print(f"  输入视频形状: {video.shape}")
        print(f"  输入视频长度: {video_len}")
        
        try:
            # 尝试前向传播
            print("  开始前向传播...")
            outputs = model(video, video_len, is_train=True)
            print("  ✅ 前向传播成功!")
            
            # 检查输出形状
            for i, output in enumerate(outputs[:-3]):  # 排除最后的None值
                if output is not None:
                    print(f"  输出 {i} 形状: {output.shape}")
                    print(f"  输出 {i} 数据类型: {output.dtype}")
                    if np.any(np.isnan(output.asnumpy())):
                        print(f"  ❌ 输出 {i} 包含NaN值!")
                    if np.any(np.isinf(output.asnumpy())):
                        print(f"  ❌ 输出 {i} 包含无穷值!")
            
        except Exception as e:
            print(f"  ❌ 前向传播失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 尝试更详细的调试
            print("\n  开始详细调试...")
            try:
                # 检查输入到卷积层的数据
                batch, temp, channel, height, width = video.shape
                inputs = video.reshape(batch * temp, channel, height, width)
                print(f"  重塑后输入形状: {inputs.shape}")
                
                # 测试ResNet骨干网络
                conv2d = model.conv2d
                print("  测试ResNet骨干网络...")
                features = conv2d(inputs[:1])  # 只测试第一帧
                print(f"  ResNet输出形状: {features.shape}")
                
            except Exception as e2:
                print(f"  详细调试也失败: {e2}")
                import traceback
                traceback.print_exc()
        
        break  # 只测试第一个样本

if __name__ == "__main__":
    try:
        debug_model_forward()
    except Exception as e:
        print(f"调试脚本失败: {e}")
        import traceback
        traceback.print_exc()
