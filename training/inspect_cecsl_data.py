#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查CE-CSL数据的实际尺寸
"""

import numpy as np
from pathlib import Path
import json
import csv

def inspect_data():
    """检查数据尺寸"""
    data_root = Path("../data/CE-CSL")
    
    # 检查是否存在清理后的词汇表
    cleaned_vocab_file = data_root / "cleaned_vocab.json"
    if cleaned_vocab_file.exists():
        print("=== 清理后的词汇表 ===")
        with open(cleaned_vocab_file, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        print(f"清理后词汇表大小: {vocab_data['vocab_size']}")
        print(f"最常见标签: {[item[0] for item in vocab_data['most_common'][:10]]}")
    
    # 首先进行全面统计
    print("🔍 CE-CSL数据集完整性检查")
    print("=" * 60)
    
    # 1. 检查原始视频vs预处理数据的数量对比
    print("\n1. 数据量对比:")
    for split in ["train", "dev", "test"]:
        # 原始视频计数
        video_dir = data_root / "video" / split
        if video_dir.exists():
            total_videos = 0
            for translator_dir in video_dir.iterdir():
                if translator_dir.is_dir():
                    videos = list(translator_dir.glob("*.mp4"))
                    total_videos += len(videos)
        else:
            total_videos = 0
        
        # 预处理数据计数
        processed_dir = data_root / "processed" / split
        if processed_dir.exists():
            processed_count = len(list(processed_dir.glob("*_frames.npy")))
        else:
            processed_count = 0
        
        # Corpus记录计数
        corpus_file = data_root / f"{split}.corpus.csv"
        if corpus_file.exists():
            with open(corpus_file, 'r', encoding='utf-8') as f:
                corpus_count = len(f.readlines()) - 1  # 减去标题行
        else:
            corpus_count = 0
        
        print(f"  {split:5}: 原始视频={total_videos:3d}, 预处理={processed_count:3d}, Corpus={corpus_count:3d}")
        
        if total_videos > processed_count:
            print(f"    ⚠️  {split} 存在未处理的视频 ({total_videos - processed_count} 个)")
    
    # 2. 检查corpus文件
    print("\n2. 检查 Corpus 文件")
    for split in ["train", "dev", "test"]:
        corpus_file = data_root / f"{split}.corpus.csv"
        if corpus_file.exists():
            with open(corpus_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                print(f"{split}.corpus.csv: {len(rows)} 条记录")
                if rows:
                    print(f"  示例: {rows[0]}")
    
    # 3. 检查元数据
    print("\n3. 检查元数据文件")
    for split in ["train", "dev"]:
        metadata_file = data_root / "processed" / split / f"{split}_metadata.json"
        print(f"检查元数据文件: {metadata_file}")
        
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            print(f"元数据条目数: {len(metadata)}")
            if metadata:
                print(f"第一个条目: {metadata[0]}")
        
        # 检查实际的帧数据
        train_dir = data_root / "processed" / split
        print(f"\n检查{split}数据目录: {train_dir}")
        
        # 找到所有.npy文件
        npy_files = list(train_dir.glob("*_frames.npy"))
        print(f"找到 {len(npy_files)} 个帧数据文件")
        
        if npy_files:
            # 检查前几个文件的尺寸
            for i, npy_file in enumerate(npy_files[:3]):
                print(f"\n文件 {i+1}: {npy_file.name}")
                try:
                    frames = np.load(npy_file)
                    print(f"  形状: {frames.shape}")
                    print(f"  数据类型: {frames.dtype}")
                    print(f"  值范围: [{frames.min():.3f}, {frames.max():.3f}]")
                    
                    # 分析数据结构
                    if len(frames.shape) == 4:  # (T, H, W, C)
                        T, H, W, C = frames.shape
                        print(f"  视频帧: T={T}, H={H}, W={W}, C={C}")
                        # 展平后的特征维度
                        feature_dim = H * W * C
                        print(f"  展平特征维度: {feature_dim}")
                    elif len(frames.shape) == 3:  # (T, H, W)
                        T, H, W = frames.shape
                        print(f"  灰度视频帧: T={T}, H={H}, W={W}")
                    elif len(frames.shape) == 2:  # (T, F)
                        T, F = frames.shape
                        print(f"  特征序列: T={T}, F={F}")
                    
                except Exception as e:
                    print(f"  加载失败: {e}")
    
    # 检查开发集
    dev_dir = data_root / "processed" / "dev"
    if dev_dir.exists():
        dev_npy_files = list(dev_dir.glob("*_frames.npy"))
        print(f"\n开发集找到 {len(dev_npy_files)} 个帧数据文件")

if __name__ == "__main__":
    inspect_data()
