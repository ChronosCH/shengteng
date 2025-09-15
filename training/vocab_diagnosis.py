#!/usr/bin/env python3
"""
词汇表诊断脚本 - 分析词汇表质量问题
"""

import json
import re
from collections import Counter

def analyze_vocabulary(vocab_path):
    """分析词汇表中的问题"""
    
    print("=== 词汇表质量诊断 ===")
    
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    word2idx = vocab_data['word2idx']
    idx2word = {v: k for k, v in word2idx.items()}
    
    print(f"词汇表总大小: {len(word2idx)}")
    
    # 1. 寻找重复的句号
    period_words = []
    for word, idx in word2idx.items():
        if '。' in word:
            period_words.append((word, idx))
    
    period_words.sort(key=lambda x: x[1])  # 按索引排序
    
    print(f"\n发现 {len(period_words)} 个包含句号的词汇:")
    for word, idx in period_words:
        # 显示空格数量
        space_count = word.count(' ')
        visible_word = word.replace(' ', '·')  # 用·显示空格
        print(f"  索引 {idx:3d}: '{visible_word}' (空格数: {space_count})")
    
    # 2. 分析重复模式
    print(f"\n=== 重复模式分析 ===")
    
    # 移除空格后的词汇统计
    stripped_words = {}
    for word, idx in word2idx.items():
        stripped = word.strip()
        if stripped in stripped_words:
            stripped_words[stripped].append((word, idx))
        else:
            stripped_words[stripped] = [(word, idx)]
    
    # 找出有重复的词汇
    duplicates = {k: v for k, v in stripped_words.items() if len(v) > 1}
    
    print(f"发现 {len(duplicates)} 个存在重复的基础词汇:")
    for base_word, variants in duplicates.items():
        if len(variants) > 2:  # 只显示重复3次以上的
            print(f"\n基础词汇 '{base_word}' 有 {len(variants)} 个变体:")
            for word, idx in sorted(variants, key=lambda x: x[1]):
                space_count = word.count(' ')
                leading_spaces = len(word) - len(word.lstrip(' '))
                trailing_spaces = len(word) - len(word.rstrip(' '))
                print(f"  索引 {idx:3d}: '{word}' (前{leading_spaces}后{trailing_spaces}空格)")
    
    # 3. 统计浪费的词汇空间
    wasted_slots = sum(len(variants) - 1 for variants in duplicates.values())
    wasted_percentage = (wasted_slots / len(word2idx)) * 100
    
    print(f"\n=== 空间浪费统计 ===")
    print(f"重复词汇浪费的词汇位: {wasted_slots}")
    print(f"浪费比例: {wasted_percentage:.2f}%")
    
    # 4. 生成清理建议
    print(f"\n=== 清理建议 ===")
    print("建议保留的词汇 (去重后):")
    
    clean_vocab = {}
    for base_word, variants in stripped_words.items():
        if base_word.strip():  # 跳过空白词汇
            # 选择最短的变体作为标准形式
            shortest = min(variants, key=lambda x: len(x[0]))
            clean_vocab[base_word] = shortest[1]
    
    print(f"清理后词汇表大小: {len(clean_vocab)} (减少 {len(word2idx) - len(clean_vocab)} 个)")
    print(f"压缩比例: {((len(word2idx) - len(clean_vocab)) / len(word2idx)) * 100:.2f}%")
    
    return duplicates, clean_vocab

def create_clean_vocabulary(original_vocab_path, clean_vocab_path):
    """创建清理后的词汇表"""
    
    print(f"\n=== 创建清理词汇表 ===")
    
    with open(original_vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    word2idx = vocab_data['word2idx']
    
    # 创建清理后的映射
    clean_word2idx = {}
    new_idx = 0
    
    # 保留空白标记作为索引0
    clean_word2idx[' '] = 0
    new_idx = 1
    
    # 去重处理
    seen_stripped = set()
    for word, old_idx in sorted(word2idx.items(), key=lambda x: x[1]):
        stripped = word.strip()
        
        # 跳过已处理的词汇和空白标记
        if stripped in seen_stripped or word == ' ':
            continue
            
        if stripped:  # 非空词汇
            # 选择原始词汇还是去除首尾空格的版本
            # 这里选择去除首尾空格的版本以保持一致性
            clean_word = stripped
            clean_word2idx[clean_word] = new_idx
            seen_stripped.add(stripped)
            new_idx += 1
    
    # 创建新的词汇数据
    clean_vocab_data = {
        'word2idx': clean_word2idx,
        'idx2word': {v: k for k, v in clean_word2idx.items()}
    }
    
    # 保存清理后的词汇表
    with open(clean_vocab_path, 'w', encoding='utf-8') as f:
        json.dump(clean_vocab_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 清理后的词汇表已保存到: {clean_vocab_path}")
    print(f"原始大小: {len(word2idx)} -> 清理后: {len(clean_word2idx)}")
    print(f"减少: {len(word2idx) - len(clean_word2idx)} 个重复词汇")

def main():
    vocab_path = "/data/shengteng/training/output_gpu/vocabulary.json"
    clean_vocab_path = "/data/shengteng/training/output_gpu/vocabulary_cleaned.json"
    
    try:
        # 分析原始词汇表
        duplicates, clean_vocab = analyze_vocabulary(vocab_path)
        
        # 创建清理后的词汇表
        create_clean_vocabulary(vocab_path, clean_vocab_path)
        
        print(f"\n=== 验证清理效果 ===")
        print("重新分析清理后的词汇表...")
        analyze_vocabulary(clean_vocab_path)
        
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
