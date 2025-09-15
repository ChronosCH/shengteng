#!/usr/bin/env python3
"""
简化的模型测试脚本
"""

import os
import sys
import json
import numpy as np
import cv2

import mindspore as ms
from mindspore import context, load_checkpoint, load_param_into_net, Tensor

# 将当前目录添加到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tfnet_model import TFNetModel

def test_model_loading():
    """测试模型加载"""
    
    model_path = "/data/shengteng/training/models/best_model.ckpt"
    vocab_path = "/data/shengteng/training/output_gpu/vocabulary.json"
    
    print("开始测试模型加载...")
    print(f"模型路径: {model_path}")
    print(f"词汇表路径: {vocab_path}")
    
    # 设置MindSpore环境
    context.set_context(
        mode=context.GRAPH_MODE,
        device_target="CPU"
    )
    
    # 加载词汇表
    print("正在加载词汇表...")
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    word2idx = vocab_data['word2idx']
    vocab_size = len(word2idx)
    print(f"✓ 词汇表加载完成，包含 {vocab_size} 个词汇")
    
    # 创建模型 - 使用safe_gpu_config.json的参数
    print("正在创建模型...")
    model = TFNetModel(
        hidden_size=128,  # 来自safe_gpu_config.json
        word_set_num=vocab_size,
        device_target="CPU",
        dataset_name="CE-CSL"
    )
    
    # 加载模型参数
    print("正在加载模型参数...")
    try:
        param_dict = load_checkpoint(model_path)
        print(f"✓ 检查点加载成功，包含 {len(param_dict)} 个参数")
        
        # 显示一些参数信息
        param_keys = list(param_dict.keys())[:10]
        print("前10个参数:")
        for key in param_keys:
            print(f"  {key}: {param_dict[key].shape}")
        
        # 加载参数到模型
        load_param_into_net(model, param_dict)
        print("✓ 模型参数加载成功")
        
        # 设置为评估模式
        model.set_train(False)
        print("✓ 模型设置为评估模式")
        
        return model, word2idx
        
    except Exception as e:
        print(f"✗ 模型加载失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def test_simple_inference(model, word2idx):
    """测试简单推理"""
    if model is None:
        print("模型未加载，跳过推理测试")
        return
    
    print("\n开始测试模型推理...")
    
    # 创建假的输入数据
    batch_size = 1
    seq_length = 10  # 较短的序列用于测试
    channels = 3
    height = 160
    width = 160
    
    # 随机输入数据
    fake_input = np.random.rand(batch_size, seq_length, channels, height, width).astype(np.float32)
    fake_length = [seq_length]
    
    print(f"输入形状: {fake_input.shape}")
    print(f"序列长度: {fake_length}")
    
    try:
        # 转换为MindSpore张量
        input_tensor = Tensor(fake_input, ms.float32)
        length_tensor = Tensor(fake_length, ms.int32)
        
        # 模型推理
        print("正在进行模型推理...")
        # MindSpore在评估模式下默认不计算梯度
        outputs = model(input_tensor, length_tensor, is_train=False)
        
        print(f"✓ 推理成功!")
        print(f"输出数量: {len(outputs)}")
        
        # 检查主要输出
        main_output = outputs[0]  # 主要输出用于推理
        print(f"主输出形状: {main_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ 推理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    try:
        # 测试模型加载
        model, word2idx = test_model_loading()
        
        # 测试简单推理
        inference_success = test_simple_inference(model, word2idx)
        
        if inference_success:
            print("\n🎉 所有测试通过! 模型加载和推理正常")
        else:
            print("\n❌ 推理测试失败")
            
    except Exception as e:
        print(f"测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
