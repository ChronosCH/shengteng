#!/usr/bin/env python3
"""
TFNet训练的简单启动脚本
"""

import os
import sys
import subprocess
import argparse
import platform
from pathlib import Path

# 将当前目录添加到路径以便导入
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils import (
        normalize_path, check_file_exists, check_directory_exists,
        print_error_details, validate_dataset_structure, print_dataset_validation
    )
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    print("Warning: Utils module not available, using basic error handling")

def check_environment():
    """检查环境是否正确设置"""
    print("=" * 60)
    print("ENVIRONMENT CHECK")
    print("=" * 60)

    all_checks_passed = True

    # Platform information
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print(f"Working Directory: {os.getcwd()}")
    print()

    # Check if we're in the correct directory
    print("Checking project structure...")
    if not os.path.exists("training"):
        print("✗ Error: Please run this script from the project root directory")
        print(f"  Current directory: {os.getcwd()}")
        print("  Expected to find: training/ folder")
        all_checks_passed = False
    else:
        print("✓ Project structure OK")

    # Check training files
    training_files = [
        "training/train_tfnet.py",
        "training/evaluator.py",
        "training/config_manager.py",
        "training/configs/tfnet_config.json"
    ]

    print("\nChecking training files...")
    for file_path in training_files:
        if UTILS_AVAILABLE:
            if not check_file_exists(file_path, os.path.basename(file_path)):
                all_checks_passed = False
        else:
            if os.path.exists(file_path):
                print(f"✓ {file_path}")
            else:
                print(f"✗ {file_path}")
                all_checks_passed = False

    # Check dataset
    print("\nChecking dataset...")
    if UTILS_AVAILABLE:
        # Use advanced dataset validation
        validation_results = validate_dataset_structure("data/CE-CSL")
        print_dataset_validation(validation_results)
        if not validation_results['valid']:
            all_checks_passed = False
    else:
        # Basic dataset check
        data_paths = [
            "data/CE-CSL/video/train",
            "data/CE-CSL/video/dev",
            "data/CE-CSL/label/train.csv",
            "data/CE-CSL/label/dev.csv"
        ]

        for path in data_paths:
            if os.path.exists(path):
                print(f"✓ {path}")
            else:
                print(f"✗ {path}")
                all_checks_passed = False

    print("\n" + "=" * 60)
    if all_checks_passed:
        print("✓ ALL ENVIRONMENT CHECKS PASSED!")
    else:
        print("✗ SOME ENVIRONMENT CHECKS FAILED!")
        print("\nPlease fix the issues above before proceeding.")
    print("=" * 60)

    return all_checks_passed

def activate_conda_env():
    """Activate conda environment"""
    print("Activating conda environment 'shengteng'...")
    
    # Check if conda is available
    try:
        result = subprocess.run(['conda', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("Warning: Conda not found. Please make sure conda is installed and in PATH.")
            return False
    except FileNotFoundError:
        print("Warning: Conda not found. Please make sure conda is installed and in PATH.")
        return False
    
    # Check if environment exists
    try:
        result = subprocess.run(['conda', 'env', 'list'], capture_output=True, text=True)
        if 'shengteng' not in result.stdout:
            print("Warning: Conda environment 'shengteng' not found.")
            print("Please create the environment first or use a different environment.")
            return False
    except Exception as e:
        print(f"Warning: Could not check conda environments: {e}")
        return False
    
    print("Conda environment check passed!")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("Checking dependencies...")
    
    required_packages = ['mindspore', 'opencv-python', 'numpy']
    
    try:
        import mindspore
        import cv2
        import numpy
        print("All required packages are available!")
        return True
    except ImportError as e:
        print(f"Missing package: {e}")
        print("Please install required packages:")
        print("  conda activate shengteng")
        print("  pip install mindspore opencv-python numpy")
        return False

def run_training(config_path=None, resume_path=None):
    """Run the training script"""
    print("Starting TFNet training...")
    
    # Build command
    cmd = [sys.executable, "training/train_tfnet.py"]
    
    if config_path:
        cmd.extend(["--config", config_path])
    
    if resume_path:
        cmd.extend(["--resume", resume_path])
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run training
        subprocess.run(cmd, check=True)
        print("Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("Training interrupted by user")
        return False

def run_evaluation(config_path=None, model_path=None):
    """Run the evaluation script"""
    print("Starting TFNet evaluation...")
    
    # Build command
    cmd = [sys.executable, "training/evaluator.py"]
    
    if config_path:
        cmd.extend(["--config", config_path])
    
    if model_path:
        cmd.extend(["--model", model_path])
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run evaluation
        subprocess.run(cmd, check=True)
        print("Evaluation completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed with exit code {e.returncode}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='TFNet Training Startup Script')
    parser.add_argument('action', choices=['train', 'eval', 'check'], 
                       help='Action to perform: train, eval, or check')
    parser.add_argument('--config', type=str, default='training/configs/tfnet_config.json',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model for evaluation')
    parser.add_argument('--skip-checks', action='store_true',
                       help='Skip environment checks')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TFNet Training System")
    print("Continuous Sign Language Recognition")
    print("=" * 60)
    
    # Environment checks
    if not args.skip_checks:
        if not check_environment():
            print("Environment check failed. Exiting.")
            sys.exit(1)
        
        if not activate_conda_env():
            print("Conda environment check failed. Continuing anyway...")
        
        if not install_dependencies():
            print("Dependency check failed. Exiting.")
            sys.exit(1)
    
    # Perform requested action
    if args.action == 'check':
        print("All checks completed successfully!")
        
    elif args.action == 'train':
        success = run_training(args.config, args.resume)
        if not success:
            sys.exit(1)
            
    elif args.action == 'eval':
        success = run_evaluation(args.config, args.model)
        if not success:
            sys.exit(1)
    
    print("=" * 60)
    print("Operation completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
