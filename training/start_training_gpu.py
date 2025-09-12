#!/usr/bin/env python3
"""
GPU优化的TFNet训练启动脚本
"""

import os
import sys
import subprocess
import argparse
import platform
from pathlib import Path

# 将当前目录添加到导入路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils import (
        normalize_path, check_file_exists, check_directory_exists,
        print_error_details, validate_dataset_structure, print_dataset_validation
    )
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    print("警告：工具模块不可用，使用基本错误处理")

def check_gpu_environment():
    """检查GPU环境是否正确设置"""
    print("=" * 60)
    print("GPU环境检查")
    print("=" * 60)

    all_checks_passed = True

    # 平台信息
    print(f"平台: {platform.system()} {platform.release()}")
    print(f"Python版本: {sys.version}")
    print(f"工作目录: {os.getcwd()}")
    print()

    # 检查conda环境
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
    print(f"Conda环境: {conda_env}")
    
    if 'mind' not in conda_env:
        print("✗ 警告: 未在mind环境中")
        print("  请运行: conda activate mind")
        all_checks_passed = False
    else:
        print("✓ 正确的conda环境已激活")
    print()

    # 检查是否在正确的目录中
    print("检查项目结构...")
    if not os.path.exists("training"):
        print("✗ 错误: 请从项目根目录运行此脚本")
        print(f"  当前目录: {os.getcwd()}")
        print("  期望找到: training/ 文件夹")
        all_checks_passed = False
    else:
        print("✓ 项目结构正常")

    # 检查GPU训练相关文件
    print("检查GPU训练文件...")
    gpu_files = [
        "train_tfnet_gpu.py",
        "configs/gpu_config.json",
        "tfnet_model.py",
        "config_manager.py",
        "data_processor.py"
    ]

    for file_path in gpu_files:
        if UTILS_AVAILABLE:
            if not check_file_exists(file_path, os.path.basename(file_path)):
                all_checks_passed = False
        else:
            if not os.path.exists(file_path):
                print(f"✗ 缺失文件: {file_path}")
                all_checks_passed = False
            else:
                print(f"✓ 找到文件: {file_path}")

    print()

    # 检查MindSpore GPU安装
    print("检查MindSpore GPU安装...")
    try:
        import mindspore as ms
        print(f"✓ MindSpore版本: {ms.__version__}")
        
        # 检查GPU是否可用
        from mindspore import context
        context.set_context(device_target="GPU")
        
        # 测试GPU张量操作
        test_tensor = ms.Tensor([[1.0, 2.0], [3.0, 4.0]], ms.float32)
        result = test_tensor + 1
        print("✓ GPU可访问且功能正常")
        
    except ImportError:
        print("✗ MindSpore未安装")
        print("  请安装MindSpore GPU版本")
        all_checks_passed = False
    except Exception as e:
        print(f"✗ GPU不可用: {e}")
        print("  请检查:")
        print("    - NVIDIA GPU驱动程序")
        print("    - CUDA安装")
        print("    - MindSpore GPU版本")
        all_checks_passed = False

    print()

    # 检查NVIDIA GPU
    print("检查NVIDIA GPU...")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ 检测到NVIDIA GPU:")
            # 显示nvidia-smi输出的前几行
            lines = result.stdout.strip().split('\n')
            for line in lines[:10]:  # 显示前10行
                if line.strip():
                    print(f"  {line}")
        else:
            print("✗ nvidia-smi执行失败")
            all_checks_passed = False
    except subprocess.TimeoutExpired:
        print("✗ nvidia-smi超时")
        all_checks_passed = False
    except FileNotFoundError:
        print("✗ 未找到nvidia-smi")
        print("  请安装NVIDIA GPU驱动程序")
        all_checks_passed = False

    print()

    # 检查数据集
    print("检查数据集...")
    if UTILS_AVAILABLE:
        dataset_paths = [
            "data/CE-CSL/video/train",
            "data/CE-CSL/video/dev",
            "data/CE-CSL/label/train.csv",
            "data/CE-CSL/label/dev.csv"
        ]

        dataset_ok = True
        for path in dataset_paths:
            if os.path.isdir(path):
                if not check_directory_exists(path, os.path.basename(path)):
                    dataset_ok = False
            else:
                if not check_file_exists(path, os.path.basename(path)):
                    dataset_ok = False

        if not dataset_ok:
            print("✗ 数据集验证失败")
            all_checks_passed = False
        else:
            print("✓ 数据集结构验证通过")
    else:
        print("? 数据集验证已跳过（工具不可用）")

    print()

    return all_checks_passed

def check_disk_space():
    """检查可用磁盘空间"""
    print("检查磁盘空间...")
    try:
        # 检查当前目录磁盘空间
        statvfs = os.statvfs('.')
        free_space_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
        
        print(f"可用磁盘空间: {free_space_gb:.1f} GB")
        
        if free_space_gb < 10:
            print("✗ 警告: 磁盘空间不足 (< 10 GB)")
            print("  训练可能因空间不足而失败:")
            print("    - 模型检查点")
            print("    - 日志文件")
            print("    - 临时文件")
            return False
        else:
            print("✓ 磁盘空间充足")
            return True
            
    except Exception as e:
        print(f"? 无法检查磁盘空间: {e}")
        return True  # 如果无法检查则不失败

def create_gpu_directories():
    """为GPU训练创建必要的目录"""
    print("创建GPU训练目录...")
    
    directories = [
        "checkpoints_gpu",
        "logs_gpu", 
        "output_gpu",
        "graphs"
    ]
    
    all_created = True
    
    for dir_path in directories:
        try:
            os.makedirs(dir_path, exist_ok=True)
            print(f"✓ 目录就绪: {dir_path}")
        except Exception as e:
            print(f"✗ 创建目录失败 {dir_path}: {e}")
            all_created = False
    
    return all_created

def run_gpu_training(config_path=None, dry_run=False):
    """运行GPU优化训练"""
    print("=" * 60)
    print("开始GPU训练")
    print("=" * 60)
    
    # 我们已经在训练目录中
    # 无需更改目录
    
    # 准备命令
    cmd = [sys.executable, "train_tfnet_gpu.py"]
    
    if config_path:
        cmd.extend(["--config", config_path])
    
    print(f"命令: {' '.join(cmd)}")
    print(f"工作目录: {os.getcwd()}")
    print()
    
    if dry_run:
        print("干运行模式 - 将执行上述命令")
        return True
    
    try:
        # 设置环境变量以提高GPU性能
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0'  # 使用第一个GPU
        env['MINDSPORE_HCCL_CONF_FILE'] = ''  # 清除任何分布式设置
        
        # 运行训练
        result = subprocess.run(cmd, env=env)
        
        if result.returncode == 0:
            print("训练成功完成！")
            return True
        else:
            print(f"训练失败，返回代码: {result.returncode}")
            return False
            
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        return False
    except Exception as e:
        print(f"训练失败，错误: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="GPU优化的TFNet训练启动器")
    parser.add_argument("--config", type=str, default="configs/gpu_config.json",
                       help="GPU配置文件路径")
    parser.add_argument("--skip-checks", action="store_true",
                       help="跳过环境检查")
    parser.add_argument("--dry-run", action="store_true",
                       help="显示将要执行的内容而不实际运行")
    parser.add_argument("--force", action="store_true",
                       help="即使检查失败也强制训练")
    
    args = parser.parse_args()
    
    print("GPU优化的TFNet训练启动器")
    print("=" * 60)
    
    # 除非跳过，否则检查环境
    if not args.skip_checks:
        print("运行环境检查...")
        
        gpu_check_passed = check_gpu_environment()
        disk_check_passed = check_disk_space()
        dir_creation_passed = create_gpu_directories()
        
        all_checks_passed = gpu_check_passed and disk_check_passed and dir_creation_passed
        
        if not all_checks_passed and not args.force:
            print("\n" + "=" * 60)
            print("环境检查失败")
            print("=" * 60)
            print("某些检查失败。请修复上述问题。")
            print("或使用 --force 强制执行（不推荐）。")
            print("或使用 --skip-checks 跳过所有检查。")
            return 1
        
        if not all_checks_passed and args.force:
            print("\n" + "=" * 60)
            print("警告: 尽管检查失败仍继续执行")
            print("=" * 60)
    
    # 运行训练
    success = run_gpu_training(args.config, args.dry_run)
    
    if success:
        print("\n" + "=" * 60)
        print("成功: GPU训练完成")
        print("=" * 60)
        return 0
    else:
        print("\n" + "=" * 60)
        print("失败: GPU训练失败")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    exit(main())
