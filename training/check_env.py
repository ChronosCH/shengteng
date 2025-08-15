#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境验证脚本
检查训练环境是否正确配置
"""

import sys
import importlib
from pathlib import Path

def check_package(package_name, import_name=None):
    """检查包是否可用"""
    import_name = import_name or package_name
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name} 已安装")
        return True
    except ImportError:
        print(f"❌ {package_name} 未安装")
        return False

def check_data_structure():
    """检查数据结构"""
    data_root = Path("../data/CE-CSL")
    
    print(f"\n📁 检查数据目录: {data_root}")
    
    if not data_root.exists():
        print(f"❌ 数据目录不存在: {data_root}")
        return False
    
    required_files = [
        "train.corpus.csv",
        "dev.corpus.csv",
        "processed/train/train_metadata.json"
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = data_root / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            all_exist = False
    
    return all_exist

def check_training_files():
    """检查训练文件"""
    print(f"\n🔧 检查训练文件:")
    
    required_files = [
        "cecsl_real_trainer.py",
        "cecsl_data_processor.py",
        "config_loader.py"
    ]
    
    all_exist = True
    for file_name in required_files:
        if Path(file_name).exists():
            print(f"✅ {file_name}")
        else:
            print(f"❌ {file_name}")
            all_exist = False
    
    return all_exist

def main():
    """主检查函数"""
    print("🔍 CE-CSL训练环境验证")
    print("=" * 40)
    
    # 检查Python版本
    print(f"🐍 Python版本: {sys.version}")
    
    # 检查必要的包
    print(f"\n📦 检查依赖包:")
    packages_ok = True
    
    required_packages = [
        ("mindspore", "mindspore"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("pathlib", "pathlib")
    ]
    
    for package_name, import_name in required_packages:
        if not check_package(package_name, import_name):
            packages_ok = False
    
    # 检查训练文件
    files_ok = check_training_files()
    
    # 检查数据结构
    data_ok = check_data_structure()
    
    # 总结
    print(f"\n📋 验证总结:")
    print(f"依赖包: {'✅' if packages_ok else '❌'}")
    print(f"训练文件: {'✅' if files_ok else '❌'}")
    print(f"数据结构: {'✅' if data_ok else '❌'}")
    
    if packages_ok and files_ok and data_ok:
        print(f"\n🎉 环境验证成功！可以开始训练")
        print(f"💡 运行训练命令: python train.py")
        return True
    else:
        print(f"\n⚠️  环境验证失败，请修复上述问题后再试")
        
        if not packages_ok:
            print(f"\n📦 安装缺失的包:")
            print(f"pip install mindspore pandas numpy")
        
        if not data_ok:
            print(f"\n📁 请确保CE-CSL数据集正确放置在 ../data/CE-CSL 目录")
        
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
