#!/usr/bin/env python3
"""
数据集调试脚本 - 检查数据集结构和路径
"""

import os
import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_manager import ConfigManager
from utils import validate_dataset_structure, print_dataset_validation, check_file_exists, check_directory_exists

def debug_dataset_structure():
    """调试数据集结构问题"""
    print("="*60)
    print("数据集结构调试")
    print("="*60)
    
    # 1. 检查当前工作目录
    print(f"当前工作目录: {os.getcwd()}")
    print(f"脚本位置: {os.path.abspath(__file__)}")
    print()
    
    # 2. 加载配置
    config_path = "configs/gpu_config.json"
    print(f"加载配置文件: {config_path}")
    
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        print("尝试使用默认配置...")
        config_manager = ConfigManager()
    else:
        print(f"✅ 配置文件存在")
        config_manager = ConfigManager(config_path)
    
    dataset_config = config_manager.get_dataset_config()
    print(f"数据集配置: {dataset_config}")
    print()
    
    # 3. 检查配置中的路径
    print("检查配置中的数据集路径...")
    paths_to_check = [
        ('train_data_path', '训练数据目录'),
        ('train_label_path', '训练标签文件'),
        ('valid_data_path', '验证数据目录'),
        ('valid_label_path', '验证标签文件'),
        ('test_data_path', '测试数据目录'),
        ('test_label_path', '测试标签文件')
    ]
    
    all_paths_valid = True
    
    for path_key, description in paths_to_check:
        path = dataset_config.get(path_key)
        print(f"\n{description} ({path_key}):")
        print(f"  配置路径: {path}")
        
        if not path:
            print(f"  ❌ 路径未配置")
            all_paths_valid = False
            continue
            
        abs_path = os.path.abspath(path)
        print(f"  绝对路径: {abs_path}")
        print(f"  路径存在: {os.path.exists(abs_path)}")
        
        if 'label' in path_key:
            # 标签文件
            if check_file_exists(path, description):
                print(f"  ✅ 标签文件验证通过")
            else:
                print(f"  ❌ 标签文件验证失败")
                all_paths_valid = False
        else:
            # 数据目录
            if check_directory_exists(path, description):
                print(f"  ✅ 数据目录验证通过")
                # 检查目录内容
                if os.path.isdir(abs_path):
                    contents = os.listdir(abs_path)
                    print(f"  目录内容数量: {len(contents)}")
                    if len(contents) > 0:
                        print(f"  前5个项目: {contents[:5]}")
            else:
                print(f"  ❌ 数据目录验证失败")
                all_paths_valid = False
    
    print("\n" + "="*60)
    
    # 4. 检查标准CE-CSL数据集结构
    print("检查标准CE-CSL数据集结构...")
    ce_csl_base_paths = [
        "data/CE-CSL",
        "../data/CE-CSL",
        "../../data/CE-CSL",
        "/data/CE-CSL"
    ]
    
    found_ce_csl = False
    for base_path in ce_csl_base_paths:
        abs_base = os.path.abspath(base_path)
        print(f"\n检查: {abs_base}")
        if os.path.exists(abs_base):
            print(f"  ✅ 路径存在")
            found_ce_csl = True
            
            # 验证CE-CSL结构
            validation_results = validate_dataset_structure(abs_base)
            print_dataset_validation(validation_results)
            
            if validation_results['valid']:
                print(f"  ✅ CE-CSL数据集结构完整")
                break
            else:
                print(f"  ⚠️  CE-CSL数据集结构不完整")
        else:
            print(f"  ❌ 路径不存在")
    
    if not found_ce_csl:
        print(f"\n❌ 在常见位置未找到CE-CSL数据集")
    
    print("\n" + "="*60)
    print("调试总结:")
    print(f"配置路径验证: {'✅ 通过' if all_paths_valid else '❌ 失败'}")
    print(f"CE-CSL数据集: {'✅ 找到' if found_ce_csl else '❌ 未找到'}")
    
    # 5. 提供修复建议
    print("\n修复建议:")
    if not all_paths_valid:
        print("1. 检查配置文件中的路径设置是否正确")
        print("2. 确保数据集文件已正确下载和解压")
        print("3. 检查当前工作目录是否正确")
    
    if not found_ce_csl:
        print("4. 下载CE-CSL数据集到 data/CE-CSL/ 目录")
        print("5. 确保目录结构如下:")
        print("   data/CE-CSL/")
        print("   ├── video/")
        print("   │   ├── train/")
        print("   │   ├── dev/")
        print("   │   └── test/")
        print("   └── label/")
        print("       ├── train.csv")
        print("       ├── dev.csv")
        print("       └── test.csv")
    
    return all_paths_valid and found_ce_csl

def create_mock_dataset():
    """创建模拟数据集用于测试"""
    print("\n" + "="*60)
    print("创建模拟数据集结构")
    print("="*60)
    
    base_path = "data/CE-CSL"
    
    # 创建目录结构
    dirs_to_create = [
        "data/CE-CSL/video/train",
        "data/CE-CSL/video/dev", 
        "data/CE-CSL/video/test",
        "data/CE-CSL/label"
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ 创建目录: {dir_path}")
    
    # 创建模拟标签文件
    label_files = [
        "data/CE-CSL/label/train.csv",
        "data/CE-CSL/label/dev.csv",
        "data/CE-CSL/label/test.csv"
    ]
    
    sample_csv_content = """video_name,translator,start_frame,label
sample_01,A,0,你好/世界
sample_02,B,0,测试/数据
"""
    
    for label_file in label_files:
        with open(label_file, 'w', encoding='utf-8') as f:
            f.write(sample_csv_content)
        print(f"✅ 创建标签文件: {label_file}")
    
    # 创建一些模拟视频目录
    sample_videos = [
        "data/CE-CSL/video/train/A/sample_01",
        "data/CE-CSL/video/train/B/sample_02", 
        "data/CE-CSL/video/dev/A/sample_01",
        "data/CE-CSL/video/test/A/sample_01"
    ]
    
    for video_dir in sample_videos:
        os.makedirs(video_dir, exist_ok=True)
        # 创建一个模拟帧文件
        frame_file = os.path.join(video_dir, "frame_001.jpg")
        # 创建一个小的占位符文件
        with open(frame_file, 'w') as f:
            f.write("mock_image_data")
        print(f"✅ 创建模拟视频: {video_dir}")
    
    print(f"\n✅ 模拟数据集创建完成: {base_path}")

def main():
    """主函数"""
    print("CE-CSL数据集调试工具")
    
    # 1. 调试当前数据集状态
    is_valid = debug_dataset_structure()
    
    # 2. 如果验证失败，询问是否创建模拟数据集
    if not is_valid:
        print(f"\n当前数据集验证失败。")
        response = input("是否创建模拟数据集用于测试? (y/n): ").strip().lower()
        
        if response == 'y':
            create_mock_dataset()
            print(f"\n重新验证数据集...")
            debug_dataset_structure()
        else:
            print(f"\n请手动修复数据集问题后重试。")
    else:
        print(f"\n✅ 数据集验证通过！")

if __name__ == "__main__":
    main()