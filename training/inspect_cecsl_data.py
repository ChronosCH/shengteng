#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查CE-CSL数据的实际尺寸
"""

import numpy as np
from pathlib import Path
import json

def inspect_data():
    """检查数据尺寸"""
    data_root = Path("../data/CE-CSL")
    
    # 检查元数据
    metadata_file = data_root / "processed" / "train" / "train_metadata.json"
    print(f"检查元数据文件: {metadata_file}")
    
    if metadata_file.exists():
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        print(f"元数据条目数: {len(metadata)}")
        if metadata:
            print(f"第一个条目: {metadata[0]}")
    
    # 检查实际的帧数据
    train_dir = data_root / "processed" / "train"
    print(f"\n检查训练数据目录: {train_dir}")
    
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
                
                # 计算展平后的特征数
                if len(frames.shape) == 4:  # (seq, h, w, c)
                    features = frames.shape[1] * frames.shape[2] * frames.shape[3]
                    print(f"  展平特征数: {features}")
                
            except Exception as e:
                print(f"  加载失败: {e}")
    
    # 检查开发集
    dev_dir = data_root / "processed" / "dev"
    if dev_dir.exists():
        dev_npy_files = list(dev_dir.glob("*_frames.npy"))
        print(f"\n开发集找到 {len(dev_npy_files)} 个帧数据文件")

if __name__ == "__main__":
    inspect_data()
