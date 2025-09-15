#!/usr/bin/env python3
"""
深度诊断脚本 - 分析模型内部行为
"""

import os
import sys
import json
import numpy as np
import cv2

import mindspore as ms
from mindspore import context, load_checkpoint, load_param_into_net, Tensor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tfnet_model import TFNetModel

def analyze_model_weights(model_path, vocab_path):
    """分析模型权重分布"""
    
    print("=== 模型权重分析 ===")
    
    # 加载参数
    param_dict = load_checkpoint(model_path)
    
    print(f"模型参数总数: {len(param_dict)}")
    
    # 分析权重统计
    for name, param in param_dict.items():
        if 'weight' in name.lower():
            weights = param.asnumpy()
            print(f"{name}:")
            print(f"  形状: {weights.shape}")
            print(f"  均值: {weights.mean():.6f}")
            print(f"  标准差: {weights.std():.6f}")
            print(f"  最小值: {weights.min():.6f}")
            print(f"  最大值: {weights.max():.6f}")
            print()

def analyze_model_outputs(model_path, vocab_path):
    """分析模型输出分布"""
    
    print("=== 模型输出分析 ===")
    
    # 设置环境
    context.set_context(
        mode=context.GRAPH_MODE,
        device_target="CPU"
    )
    
    # 加载词汇表
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    word2idx = vocab_data['word2idx']
    idx2word = {v: k for k, v in word2idx.items()}
    vocab_size = len(word2idx)
    
    # 创建模型
    model = TFNetModel(
        hidden_size=128,
        word_set_num=vocab_size,
        device_target="CPU",
        dataset_name="CE-CSL"
    )
    
    # 加载参数
    param_dict = load_checkpoint(model_path)
    load_param_into_net(model, param_dict)
    model.set_train(False)
    
    # 创建多个不同的测试输入
    test_cases = [
        {"name": "随机输入", "input": np.random.rand(1, 10, 3, 160, 160).astype(np.float32)},
        {"name": "全零输入", "input": np.zeros((1, 10, 3, 160, 160), dtype=np.float32)},
        {"name": "全一输入", "input": np.ones((1, 10, 3, 160, 160), dtype=np.float32)},
    ]
    
    for case in test_cases:
        print(f"\n--- {case['name']} ---")
        
        input_tensor = Tensor(case['input'], ms.float32)
        length_tensor = Tensor([10], ms.int32)
        
        # 模型推理
        outputs = model(input_tensor, length_tensor, is_train=False)
        
        # 分析所有输出
        for i, output in enumerate(outputs):
            if output is not None:
                output_np = output.asnumpy()
                print(f"输出 {i} 形状: {output_np.shape}")
                
                if len(output_np.shape) == 3:  # (T, B, C) 格式的logits
                    # 分析logits分布
                    print(f"  Logits 统计:")
                    print(f"    均值: {output_np.mean():.6f}")
                    print(f"    标准差: {output_np.std():.6f}")
                    print(f"    最小值: {output_np.min():.6f}")
                    print(f"    最大值: {output_np.max():.6f}")
                    
                    # 分析每个时间步的最大预测
                    seq_len = output_np.shape[0]
                    predictions = []
                    for t in range(min(seq_len, 5)):  # 只看前5个时间步
                        max_idx = np.argmax(output_np[t, 0])
                        max_val = output_np[t, 0, max_idx]
                        word = idx2word.get(max_idx, 'unknown')
                        predictions.append((t, max_idx, word, max_val))
                        print(f"  时间步 {t}: 索引={max_idx}, 词='{word}', 值={max_val:.4f}")
                    
                    # 分析词汇预测分布
                    all_max_indices = []
                    for t in range(seq_len):
                        max_idx = np.argmax(output_np[t, 0])
                        all_max_indices.append(max_idx)
                    
                    unique_indices, counts = np.unique(all_max_indices, return_counts=True)
                    print(f"  预测词汇分布:")
                    for idx, count in zip(unique_indices[:5], counts[:5]):  # 前5个最常见的
                        word = idx2word.get(idx, 'unknown')
                        print(f"    索引 {idx} ('{word}'): {count} 次")
                
                elif len(output_np.shape) == 1:  # 长度信息
                    print(f"  长度信息: {output_np}")

def check_vocabulary_distribution(vocab_path):
    """检查词汇表分布"""
    
    print("=== 词汇表分析 ===")
    
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    word2idx = vocab_data['word2idx']
    
    print(f"词汇表大小: {len(word2idx)}")
    print(f"索引0对应词汇: '{list(word2idx.keys())[0]}'  (blank token)")
    
    # 查找"义务"的索引
    for word, idx in word2idx.items():
        if '义务' in word:
            print(f"找到'义务': 索引={idx}, 完整词汇='{word}'")
    
    # 显示一些词汇样例
    print("\n词汇样例 (前20个):")
    for i, (word, idx) in enumerate(list(word2idx.items())[:20]):
        print(f"  {idx}: '{word}'")

def diagnose_ctc_issue(model_path, vocab_path):
    """诊断CTC相关问题"""
    
    print("=== CTC诊断 ===")
    
    # 设置环境
    context.set_context(
        mode=context.GRAPH_MODE,
        device_target="CPU"
    )
    
    # 加载词汇表
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    word2idx = vocab_data['word2idx']
    idx2word = {v: k for k, v in word2idx.items()}
    vocab_size = len(word2idx)
    
    # 创建模型
    model = TFNetModel(
        hidden_size=128,
        word_set_num=vocab_size,
        device_target="CPU",
        dataset_name="CE-CSL"
    )
    
    # 加载参数
    param_dict = load_checkpoint(model_path)
    load_param_into_net(model, param_dict)
    model.set_train(False)
    
    # 测试输入
    input_tensor = Tensor(np.random.rand(1, 20, 3, 160, 160).astype(np.float32), ms.float32)
    length_tensor = Tensor([20], ms.int32)
    
    outputs = model(input_tensor, length_tensor, is_train=False)
    
    print(f"模型输出数量: {len(outputs)}")
    for i, output in enumerate(outputs):
        if output is not None:
            print(f"输出 {i}: 形状={output.shape}")
    
    # 重点分析主要输出 (通常是outputs[0])
    main_logits = outputs[0].asnumpy()  # (T, B, C)
    print(f"\n主要logits形状: {main_logits.shape}")
    
    # 检查blank token的概率
    blank_probs = main_logits[:, 0, 0]  # 所有时间步的blank概率
    print(f"Blank token logits: min={blank_probs.min():.4f}, max={blank_probs.max():.4f}, mean={blank_probs.mean():.4f}")
    
    # 检查其他token的概率范围
    non_blank_probs = main_logits[:, 0, 1:]  # 所有非blank token
    print(f"Non-blank logits: min={non_blank_probs.min():.4f}, max={non_blank_probs.max():.4f}, mean={non_blank_probs.mean():.4f}")
    
    # 检查softmax后的概率
    from scipy.special import softmax
    softmax_probs = softmax(main_logits, axis=-1)
    blank_softmax = softmax_probs[:, 0, 0]
    print(f"Blank softmax概率: min={blank_softmax.min():.4f}, max={blank_softmax.max():.4f}, mean={blank_softmax.mean():.4f}")

def main():
    model_path = "/data/shengteng/training/models/best_model.ckpt"
    vocab_path = "/data/shengteng/training/output_gpu/vocabulary.json"
    
    try:
        check_vocabulary_distribution(vocab_path)
        print("\n" + "="*60)
        analyze_model_weights(model_path, vocab_path)
        print("\n" + "="*60)
        analyze_model_outputs(model_path, vocab_path)
        print("\n" + "="*60)
        diagnose_ctc_issue(model_path, vocab_path)
        
    except Exception as e:
        print(f"诊断过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
