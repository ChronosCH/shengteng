#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标签质量验证工具
验证清理后的标签质量
"""

import csv
import json
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

def validate_cleaned_labels(data_root: str = "../data/CE-CSL"):
    """验证清理后的标签质量"""
    data_path = Path(data_root)
    
    print("📊 标签质量验证")
    print("=" * 50)
    
    all_labels = []
    split_stats = {}
    
    # 收集所有标签
    for split in ['train', 'dev', 'test']:
        corpus_file = data_path / f"{split}.corpus.csv"
        if not corpus_file.exists():
            continue
            
        labels = []
        with open(corpus_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row['label']
                labels.append(label)
                all_labels.append(label)
        
        split_stats[split] = {
            'count': len(labels),
            'unique': len(set(labels)),
            'max_length': max(len(l) for l in labels) if labels else 0,
            'avg_length': sum(len(l) for l in labels) / len(labels) if labels else 0
        }
    
    # 整体统计
    label_counts = Counter(all_labels)
    
    print("📈 整体统计:")
    print(f"  总标签数: {len(all_labels)}")
    print(f"  唯一标签数: {len(label_counts)}")
    print(f"  最大长度: {max(len(l) for l in all_labels)}")
    print(f"  平均长度: {sum(len(l) for l in all_labels) / len(all_labels):.1f}")
    
    print("\n📋 各集合统计:")
    for split, stats in split_stats.items():
        print(f"  {split}: {stats['count']} 条记录, {stats['unique']} 个唯一标签, "
              f"最大长度 {stats['max_length']}, 平均长度 {stats['avg_length']:.1f}")
    
    print("\n🏆 最常见标签 (前15个):")
    for i, (label, count) in enumerate(label_counts.most_common(15), 1):
        print(f"  {i:2d}. '{label}': {count} 次")
    
    # 检查质量问题
    print("\n🔍 质量检查:")
    issues = []
    
    # 检查异常长标签
    long_labels = [label for label in all_labels if len(label) > 6]
    if long_labels:
        issues.append(f"过长标签 ({len(set(long_labels))} 个): {list(set(long_labels))[:5]}")
    
    # 检查特殊字符
    special_chars = set()
    for label in all_labels:
        for char in label:
            if not ('\u4e00' <= char <= '\u9fff' or char.isalpha() or char.isdigit()):
                special_chars.add(char)
    
    if special_chars:
        issues.append(f"特殊字符: {sorted(special_chars)}")
    
    # 检查空标签
    empty_labels = [label for label in all_labels if not label.strip()]
    if empty_labels:
        issues.append(f"空标签: {len(empty_labels)} 个")
    
    if issues:
        print("  ⚠️  发现问题:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  ✅ 未发现质量问题")
    
    # 生成分布图
    try:
        generate_label_distribution_plot(label_counts, data_path)
    except Exception as e:
        print(f"  生成分布图失败: {e}")
    
    return {
        'total_labels': len(all_labels),
        'unique_labels': len(label_counts),
        'split_stats': split_stats,
        'top_labels': label_counts.most_common(20),
        'issues': issues
    }

def generate_label_distribution_plot(label_counts: Counter, output_dir: Path):
    """生成标签分布图"""
    try:
        import matplotlib
        matplotlib.use('Agg')  # 不显示图形界面
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文
        
        # 取前20个最常见标签
        top_labels = label_counts.most_common(20)
        labels, counts = zip(*top_labels)
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(labels)), counts, alpha=0.7)
        plt.xlabel('标签')
        plt.ylabel('频次')
        plt.title('标签分布 (前20个)')
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
        
        # 添加数值标签
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plot_file = output_dir / "label_distribution.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 标签分布图已保存: {plot_file}")
        
    except Exception as e:
        print(f"  生成分布图失败: {e}")

if __name__ == "__main__":
    validate_cleaned_labels()
