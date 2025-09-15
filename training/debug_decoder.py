#!/usr/bin/env python3
"""
调试解码器的脚本
"""

import os
import sys
import json
import numpy as np

import mindspore as ms
from mindspore import context, load_checkpoint, load_param_into_net, Tensor

# 将当前目录添加到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tfnet_model import TFNetModel
from decoder import CTCDecoder

def debug_decoder():
    """调试解码器"""
    
    model_path = "/data/shengteng/training/models/best_model.ckpt"
    vocab_path = "/data/shengteng/training/output_gpu/vocabulary.json"
    
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
    
    print(f"词汇表大小: {vocab_size}")
    print(f"blank_id (空白标记): 0 -> '{idx2word.get(0, 'unknown')}'")
    
    # 创建解码器
    decoder = CTCDecoder(
        gloss_dict=word2idx,
        num_classes=vocab_size,
        search_mode='max',
        blank_id=0
    )
    
    # 创建假的模型输出来测试解码器
    seq_len = 10
    batch_size = 1
    num_classes = vocab_size
    
    # 创建假的logits - 大部分为blank，少数为有意义的类别
    fake_logits = np.random.randn(seq_len, batch_size, num_classes).astype(np.float32)
    
    # 让一些时间步有明确的预测（非blank）
    fake_logits[2, 0, 100] = 10.0  # 强制预测类别100
    fake_logits[5, 0, 200] = 10.0  # 强制预测类别200
    fake_logits[8, 0, 300] = 10.0  # 强制预测类别300
    
    # 其他时间步倾向于预测blank (index 0)
    for t in [0, 1, 3, 4, 6, 7, 9]:
        fake_logits[t, 0, 0] = 5.0
    
    fake_lengths = np.array([seq_len], dtype=np.int32)
    
    print(f"测试输入形状: {fake_logits.shape}")
    print(f"序列长度: {fake_lengths}")
    
    # 转换为MindSpore张量
    logits_tensor = Tensor(fake_logits, ms.float32)
    lengths_tensor = Tensor(fake_lengths, ms.int32)
    
    # 解码
    print("\n开始解码...")
    predictions = decoder.decode(
        nn_output=logits_tensor,
        vid_lgt=lengths_tensor,
        batch_first=False
    )
    
    print(f"解码结果: {predictions}")
    
    if predictions and len(predictions) > 0:
        pred_sequence = predictions[0]
        print(f"预测序列索引: {pred_sequence}")
        
        # 转换为单词
        predicted_words = []
        for idx in pred_sequence:
            if idx in idx2word:
                word = idx2word[idx]
                predicted_words.append(word)
                print(f"  索引 {idx} -> '{word}'")
        
        print(f"预测单词: {predicted_words}")
    else:
        print("解码结果为空!")
    
    # 检查原始logits的最大值位置
    print("\n分析原始logits:")
    for t in range(seq_len):
        max_idx = np.argmax(fake_logits[t, 0])
        max_val = fake_logits[t, 0, max_idx]
        word = idx2word.get(max_idx, 'unknown')
        print(f"时间步 {t}: 最大值索引={max_idx} ('{word}'), 值={max_val:.2f}")

def debug_real_model_output():
    """调试真实模型输出"""
    
    model_path = "/data/shengteng/training/models/best_model.ckpt"
    vocab_path = "/data/shengteng/training/output_gpu/vocabulary.json"
    
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
    
    # 创建假输入
    batch_size = 1
    seq_length = 10
    channels = 3
    height = 160
    width = 160
    
    fake_input = np.random.rand(batch_size, seq_length, channels, height, width).astype(np.float32)
    fake_length = [seq_length]
    
    input_tensor = Tensor(fake_input, ms.float32)
    length_tensor = Tensor(fake_length, ms.int32)
    
    print("使用真实模型进行推理...")
    outputs = model(input_tensor, length_tensor, is_train=False)
    
    logits = outputs[0] # (T, B, C)
    pred_lengths = outputs[5] # (B,)
    
    print(f"模型输出形状: {logits.shape}")
    print(f"预测长度: {pred_lengths}")
    
    # 转换为numpy分析
    logits_np = logits.asnumpy()
    lengths_np = pred_lengths.asnumpy()
    
    print(f"实际长度值: {lengths_np}")
    
    # 分析每个时间步的预测
    seq_len, batch_size, num_classes = logits_np.shape
    print(f"\n分析模型输出 (前10个时间步):")
    
    for t in range(min(10, seq_len)):
        max_idx = np.argmax(logits_np[t, 0])
        max_val = logits_np[t, 0, max_idx]
        word = idx2word.get(max_idx, 'unknown')
        
        # 检查blank概率
        blank_val = logits_np[t, 0, 0]
        
        print(f"时间步 {t}: 最大值索引={max_idx} ('{word}'), 值={max_val:.2f}, blank值={blank_val:.2f}")
    
    # 创建解码器并解码
    decoder = CTCDecoder(
        gloss_dict=word2idx,
        num_classes=vocab_size,
        search_mode='max',
        blank_id=0
    )
    
    print("\n使用真实模型输出进行解码...")
    predictions = decoder.decode(
        nn_output=logits,
        vid_lgt=pred_lengths,
        batch_first=False
    )
    
    print(f"解码结果: {predictions}")
    
    if predictions and len(predictions) > 0:
        pred_sequence = predictions[0]
        print(f"预测序列索引: {pred_sequence}")
        
        predicted_words = []
        for idx in pred_sequence:
            if idx in idx2word:
                word = idx2word[idx]
                predicted_words.append(word)
        
        print(f"预测单词: {predicted_words}")
    else:
        print("解码结果为空!")

def main():
    print("=== 调试假数据解码器 ===")
    debug_decoder()
    
    print("\n" + "="*50)
    print("=== 调试真实模型输出 ===")
    debug_real_model_output()

if __name__ == "__main__":
    main()
