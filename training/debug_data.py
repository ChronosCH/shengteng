#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据诊断脚本
"""

import os
import csv
import numpy as np
from pathlib import Path

def check_csv_encoding():
    """检查CSV文件编码问题"""
    data_root = Path("../data/CE-CSL")
    
    # 检查corpus文件
    for split in ["train", "dev", "test"]:
        csv_path = data_root / f"{split}.corpus.csv"
        if not csv_path.exists():
            print(f"文件不存在: {csv_path}")
            continue
            
        print(f"\n=== 检查 {split}.corpus.csv ===")
        
        # 尝试不同编码
        for encoding in ["utf-8", "utf-8-sig", "gbk", "gb2312"]:
            try:
                with open(csv_path, "r", encoding=encoding, newline="") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    print(f"编码 {encoding}: 成功读取 {len(rows)} 行")
                    if len(rows) > 0:
                        print(f"示例数据: video_id={rows[0]['video_id']}, label={rows[0]['label']}")
                        # 统计标签分布
                        labels = [row['label'] for row in rows]
                        label_counts = {}
                        for label in labels:
                            label_counts[label] = label_counts.get(label, 0) + 1
                        print(f"标签分布: {label_counts}")
                    break
            except Exception as e:
                print(f"编码 {encoding}: 失败 - {e}")
    
    # 检查label文件
    label_dir = data_root / "label"
    if label_dir.exists():
        print(f"\n=== 检查 label 目录 ===")
        for split in ["train", "dev", "test"]:
            label_file = label_dir / f"{split}.csv"
            if label_file.exists():
                print(f"找到标签文件: {label_file}")
                # 简单检查行数
                with open(label_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    print(f"  {split}.csv: {len(lines)} 行")

def check_npy_files():
    """检查npy文件"""
    data_root = Path("../data/CE-CSL/processed")
    
    for split in ["train", "dev"]:
        split_dir = data_root / split
        if not split_dir.exists():
            print(f"目录不存在: {split_dir}")
            continue
            
        print(f"\n=== 检查 {split} 特征文件 ===")
        
        npy_files = list(split_dir.glob("*_frames.npy"))
        print(f"找到 {len(npy_files)} 个特征文件")
        
        if len(npy_files) > 0:
            # 检查第一个文件
            first_file = npy_files[0]
            try:
                data = np.load(first_file)
                print(f"文件: {first_file.name}")
                print(f"形状: {data.shape}")
                print(f"数据类型: {data.dtype}")
                print(f"数据范围: [{data.min():.3f}, {data.max():.3f}]")
                
                # 检查是否需要转换为2D
                if data.ndim == 4:
                    print("4D数据，需要转换为2D")
                    if data.shape[-1] in (1, 3):  # NHWC
                        data_2d = data.mean(axis=(1, 2))
                        print(f"转换后形状: {data_2d.shape}")
                    elif data.shape[1] in (1, 3):  # NCHW  
                        data_2d = data.mean(axis=(2, 3))
                        print(f"转换后形状: {data_2d.shape}")
                
            except Exception as e:
                print(f"读取失败: {e}")

if __name__ == "__main__":
    print("🔍 开始数据诊断...")
    check_csv_encoding()
    check_npy_files()
