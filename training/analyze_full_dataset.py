#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全面分析CE-CSL数据集
检查原始视频数量vs预处理数量的差异
"""

import os
import csv
from pathlib import Path
import json

def analyze_full_dataset():
    """全面分析数据集"""
    data_root = Path("../data/CE-CSL")
    
    print("🔍 CE-CSL数据集全面分析")
    print("=" * 60)
    
    # 1. 检查原始视频数量
    print("\n1. 原始视频统计:")
    video_root = data_root / "video"
    
    for split in ["train", "dev", "test"]:
        split_dir = video_root / split
        if not split_dir.exists():
            print(f"  {split}: 目录不存在")
            continue
            
        total_videos = 0
        translators = []
        
        for translator_dir in sorted(split_dir.iterdir()):
            if translator_dir.is_dir():
                translators.append(translator_dir.name)
                videos = list(translator_dir.glob("*.mp4"))
                total_videos += len(videos)
                print(f"  {split}/{translator_dir.name}: {len(videos)} 个视频")
        
        print(f"  {split} 总计: {total_videos} 个视频，{len(translators)} 个译员")
    
    # 2. 检查预处理数据数量
    print("\n2. 预处理数据统计:")
    processed_root = data_root / "processed"
    
    for split in ["train", "dev", "test"]:
        split_dir = processed_root / split
        if not split_dir.exists():
            print(f"  {split}: 预处理目录不存在")
            continue
            
        npy_files = list(split_dir.glob("*_frames.npy"))
        metadata_file = split_dir / f"{split}_metadata.json"
        
        print(f"  {split}: {len(npy_files)} 个.npy文件")
        
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            print(f"  {split}: 元数据记录 {len(metadata)} 条")
        else:
            print(f"  {split}: 无元数据文件")
    
    # 3. 检查corpus文件
    print("\n3. Corpus文件统计:")
    
    for split in ["train", "dev", "test"]:
        corpus_file = data_root / f"{split}.corpus.csv"
        if corpus_file.exists():
            with open(corpus_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                print(f"  {split}.corpus.csv: {len(rows)} 条记录")
                
                # 分析video_id模式
                if rows:
                    video_ids = [row['video_id'] for row in rows]
                    unique_videos = len(set(video_ids))
                    print(f"    独立视频ID: {unique_videos} 个")
                    print(f"    ID示例: {video_ids[:3]}...")
        else:
            print(f"  {split}.corpus.csv: 文件不存在")
    
    # 4. 检查label文件
    print("\n4. Label文件统计:")
    label_dir = data_root / "label"
    
    if label_dir.exists():
        for split in ["train", "dev", "test"]:
            label_file = label_dir / f"{split}.csv"
            if label_file.exists():
                with open(label_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    print(f"  {split}.csv: {len(lines)-1} 条记录（含标题行）")
            else:
                print(f"  {split}.csv: 文件不存在")
    else:
        print("  label目录不存在")
    
    # 5. 识别问题
    print("\n🔧 问题诊断:")
    print("=" * 60)
    
    # 检查video_id命名是否与实际文件对应
    print("检查video_id命名规则...")
    
    for split in ["train", "dev"]:  # 重点检查有数据的split
        video_dir = video_root / split
        corpus_file = data_root / f"{split}.corpus.csv"
        
        if not video_dir.exists() or not corpus_file.exists():
            continue
            
        # 统计实际视频文件
        actual_videos = []
        for translator_dir in sorted(video_dir.iterdir()):
            if translator_dir.is_dir():
                videos = sorted(translator_dir.glob("*.mp4"))
                for i, video_file in enumerate(videos):
                    actual_videos.append({
                        'translator': translator_dir.name,
                        'file_name': video_file.name,
                        'full_path': video_file,
                        'expected_id': f"{split}_video_{len(actual_videos):03d}"
                    })
        
        # 读取corpus中的video_id
        with open(corpus_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            corpus_ids = [row['video_id'] for row in reader]
        
        print(f"\n{split} 集分析:")
        print(f"  实际视频文件: {len(actual_videos)} 个")
        print(f"  Corpus记录: {len(corpus_ids)} 个")
        print(f"  预处理文件: {len(list((processed_root / split).glob('*_frames.npy')))} 个")
        
        if len(actual_videos) > len(corpus_ids):
            print(f"  ⚠️  问题: 实际视频数量({len(actual_videos)}) > Corpus记录({len(corpus_ids)})")
            print(f"      可能需要重新生成corpus文件或重新预处理")
        
        # 显示视频文件映射示例
        if actual_videos:
            print(f"  视频文件映射示例:")
            for i, video in enumerate(actual_videos[:5]):
                print(f"    {video['expected_id']} <- {video['translator']}/{video['file_name']}")

if __name__ == "__main__":
    analyze_full_dataset()
