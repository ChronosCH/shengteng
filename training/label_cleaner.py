#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CE-CSL标签清理工具
处理异常标签、特殊字符和长标签问题
"""

import re
import json
import csv
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LabelCleaner:
    """标签清理器"""
    
    def __init__(self):
        # 特殊字符清理规则
        self.special_char_patterns = [
            (r'\{[^}]*\}', ''),  # 删除 {xxx} 注释
            (r'\([^)]*\)', ''),  # 删除 (xxx) 注释
            (r'\[[^\]]*\]', ''), # 删除 [xxx] 注释
            (r'（[^）]*）', ''),  # 删除 （xxx） 注释
            (r'[{}()\[\]（）]', ''),  # 删除剩余括号
        ]
        
        # 常见手语词汇映射
        self.common_mappings = {
            '文文章的第一个手势动作': '文章',
            '你照顾': '照顾',
            '龙虾': '龙虾',
            # 可以根据需要添加更多映射
        }
        
        # 停用词列表
        self.stop_words = {'的', '了', '是', '在', '和', '与', '或'}
        
        # 最大标签长度
        self.max_label_length = 6
    
    def clean_single_label(self, label: str) -> str:
        """清理单个标签"""
        if not label or not label.strip():
            return ""
        
        original_label = label
        cleaned = label.strip()
        
        # 1. 应用预定义映射
        if cleaned in self.common_mappings:
            cleaned = self.common_mappings[cleaned]
            logger.debug(f"映射: '{original_label}' -> '{cleaned}'")
            return cleaned
        
        # 2. 清理特殊字符和注释
        for pattern, replacement in self.special_char_patterns:
            cleaned = re.sub(pattern, replacement, cleaned)
        
        # 3. 清理多余空格
        cleaned = re.sub(r'\s+', '', cleaned)
        
        # 4. 处理过长标签
        if len(cleaned) > self.max_label_length:
            # 尝试提取主要词汇
            cleaned = self._extract_main_word(cleaned)
        
        # 5. 处理空结果
        if not cleaned:
            # 尝试从原始标签提取有用信息
            cleaned = self._fallback_extraction(original_label)
        
        logger.debug(f"清理: '{original_label}' -> '{cleaned}'")
        return cleaned
    
    def _extract_main_word(self, long_label: str) -> str:
        """从长标签中提取主要词汇"""
        # 常见的词汇优先级
        priority_words = ['你', '我', '他', '她', '谢谢', '请', '好', '不', '是', '有', '没有']
        
        for word in priority_words:
            if word in long_label:
                return word
        
        # 如果没有找到优先词汇，取前面的字符
        if len(long_label) > 2:
            return long_label[:2]
        
        return long_label
    
    def _fallback_extraction(self, original_label: str) -> str:
        """后备提取方法"""
        # 移除所有特殊字符，保留中文字符
        chinese_only = re.sub(r'[^\u4e00-\u9fff]', '', original_label)
        
        if chinese_only:
            # 返回前两个中文字符
            return chinese_only[:2] if len(chinese_only) >= 2 else chinese_only
        
        # 如果没有中文字符，返回默认标签
        return "未知"
    
    def analyze_labels(self, corpus_files: List[str]) -> Dict:
        """分析所有标签"""
        all_labels = []
        label_counts = Counter()
        problematic_labels = []
        
        for corpus_file in corpus_files:
            if not Path(corpus_file).exists():
                continue
                
            with open(corpus_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    label = row['label'].strip()
                    all_labels.append(label)
                    label_counts[label] += 1
                    
                    # 检查问题标签
                    if len(label) > self.max_label_length or any(char in label for char in '{}()[]（）'):
                        problematic_labels.append(label)
        
        return {
            'total_labels': len(all_labels),
            'unique_labels': len(label_counts),
            'label_distribution': dict(label_counts.most_common(20)),
            'problematic_labels': list(set(problematic_labels)),
            'max_length': max(len(label) for label in all_labels) if all_labels else 0,
            'avg_length': sum(len(label) for label in all_labels) / len(all_labels) if all_labels else 0
        }
    
    def clean_corpus_files(self, data_root: str) -> Dict:
        """清理所有corpus文件"""
        data_path = Path(data_root)
        results = {}
        
        for split in ['train', 'dev', 'test']:
            corpus_file = data_path / f"{split}.corpus.csv"
            if not corpus_file.exists():
                continue
            
            logger.info(f"清理 {split}.corpus.csv...")
            
            # 备份原文件
            backup_file = data_path / f"{split}.corpus.csv.backup"
            if not backup_file.exists():
                corpus_file.rename(backup_file)
                logger.info(f"备份原文件: {backup_file}")
            
            # 读取并清理数据
            cleaned_records = []
            original_records = []
            
            with open(backup_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    original_label = row['label']
                    cleaned_label = self.clean_single_label(original_label)
                    
                    # 跳过空标签
                    if not cleaned_label:
                        logger.warning(f"跳过空标签记录: {row['video_id']}")
                        continue
                    
                    cleaned_record = dict(row)
                    cleaned_record['label'] = cleaned_label
                    cleaned_records.append(cleaned_record)
                    original_records.append(original_label)
            
            # 写入清理后的文件
            with open(corpus_file, 'w', encoding='utf-8', newline='') as f:
                if cleaned_records:
                    writer = csv.DictWriter(f, fieldnames=cleaned_records[0].keys())
                    writer.writeheader()
                    writer.writerows(cleaned_records)
            
            # 统计结果
            results[split] = {
                'original_count': len(original_records),
                'cleaned_count': len(cleaned_records),
                'removed_count': len(original_records) - len(cleaned_records),
                'unique_labels': len(set(record['label'] for record in cleaned_records))
            }
            
            logger.info(f"{split} 清理完成: {results[split]}")
        
        return results
    
    def create_cleaned_vocabulary(self, data_root: str) -> Dict:
        """创建清理后的词汇表"""
        data_path = Path(data_root)
        
        # 收集所有清理后的标签
        all_labels = set()
        label_counts = Counter()
        
        for split in ['train', 'dev', 'test']:
            corpus_file = data_path / f"{split}.corpus.csv"
            if corpus_file.exists():
                with open(corpus_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        label = row['label']
                        all_labels.add(label)
                        label_counts[label] += 1
        
        # 创建词汇表
        vocab_list = ['<PAD>', '<UNK>'] + sorted(list(all_labels))
        word2idx = {word: i for i, word in enumerate(vocab_list)}
        
        vocab_info = {
            'vocab_size': len(vocab_list),
            'word2idx': word2idx,
            'idx2word': vocab_list,
            'label_distribution': dict(label_counts),
            'most_common': label_counts.most_common(10)
        }
        
        # 保存词汇表
        vocab_file = data_path / "cleaned_vocab.json"
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(vocab_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"清理后词汇表已保存: {vocab_file}")
        logger.info(f"词汇表大小: {len(vocab_list)}")
        
        return vocab_info

def main():
    print("🧹 CE-CSL标签清理工具")
    print("=" * 50)
    
    data_root = "../data/CE-CSL"
    cleaner = LabelCleaner()
    
    # 1. 分析原始标签
    print("\n1. 分析原始标签...")
    corpus_files = [
        f"{data_root}/train.corpus.csv",
        f"{data_root}/dev.corpus.csv", 
        f"{data_root}/test.corpus.csv"
    ]
    
    analysis = cleaner.analyze_labels(corpus_files)
    print(f"总标签数: {analysis['total_labels']}")
    print(f"唯一标签数: {analysis['unique_labels']}")
    print(f"最大长度: {analysis['max_length']}")
    print(f"平均长度: {analysis['avg_length']:.1f}")
    print(f"问题标签数: {len(analysis['problematic_labels'])}")
    
    # 2. 清理corpus文件
    print("\n2. 清理corpus文件...")
    results = cleaner.clean_corpus_files(data_root)
    
    # 3. 创建清理后的词汇表
    print("\n3. 创建清理后的词汇表...")
    vocab_info = cleaner.create_cleaned_vocabulary(data_root)
    
    # 4. 总结
    print("\n🎉 清理完成!")
    print("=" * 50)
    for split, result in results.items():
        print(f"{split}: {result['original_count']} -> {result['cleaned_count']} "
              f"(移除 {result['removed_count']}, 唯一标签 {result['unique_labels']})")
    
    print(f"\n最终词汇表大小: {vocab_info['vocab_size']}")
    print("最常见标签:", [item[0] for item in vocab_info['most_common']])
    
    print("\n建议运行以下命令验证结果:")
    print("python inspect_cecsl_data.py")

if __name__ == "__main__":
    main()
