#!/usr/bin/env python3
"""
GPU-Optimized startup script for TFNet training
"""

import os
import sys
import subprocess
import argparse
import platform
from pathlib import Path

# Add current directory to path for imports
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

def check_gpu_environment():
    """Check if GPU environment is properly set up"""
    print("=" * 60)
    print("GPU ENVIRONMENT CHECK")
    print("=" * 60)

    all_checks_passed = True

    # Platform information
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print(f"Working Directory: {os.getcwd()}")
    print()

    # Check conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
    print(f"Conda Environment: {conda_env}")
    
    if 'mindspore-gpu' not in conda_env:
        print("✗ Warning: Not in mindspore-gpu environment")
        print("  Please run: conda activate mindspore-gpu")
        all_checks_passed = False
    else:
        print("✓ Correct conda environment activated")
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

    # Check GPU-specific training files
    print("Checking GPU training files...")
    gpu_files = [
        "training/train_tfnet_gpu.py",
        "training/configs/gpu_config.json",
        "training/tfnet_model.py",
        "training/config_manager.py",
        "training/data_processor.py"
    ]

    for file_path in gpu_files:
        if UTILS_AVAILABLE:
            if not check_file_exists(file_path, os.path.basename(file_path)):
                all_checks_passed = False
        else:
            if not os.path.exists(file_path):
                print(f"✗ Missing: {file_path}")
                all_checks_passed = False
            else:
                print(f"✓ Found: {file_path}")

    print()

    # Check MindSpore GPU installation
    print("Checking MindSpore GPU installation...")
    try:
        import mindspore as ms
        print(f"✓ MindSpore version: {ms.__version__}")
        
        # Check if GPU is available
        from mindspore import context
        context.set_context(device_target="GPU")
        
        # Test GPU tensor operation
        test_tensor = ms.Tensor([[1.0, 2.0], [3.0, 4.0]], ms.float32)
        result = test_tensor + 1
        print("✓ GPU is accessible and functional")
        
    except ImportError:
        print("✗ MindSpore not installed")
        print("  Please install MindSpore GPU version")
        all_checks_passed = False
    except Exception as e:
        print(f"✗ GPU not available: {e}")
        print("  Please check:")
        print("    - NVIDIA GPU drivers")
        print("    - CUDA installation")
        print("    - MindSpore GPU version")
        all_checks_passed = False

    print()

    # Check NVIDIA GPU
    print("Checking NVIDIA GPU...")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ NVIDIA GPU detected:")
            # Show first few lines of nvidia-smi output
            lines = result.stdout.strip().split('\n')
            for line in lines[:10]:  # Show first 10 lines
                if line.strip():
                    print(f"  {line}")
        else:
            print("✗ nvidia-smi failed")
            all_checks_passed = False
    except subprocess.TimeoutExpired:
        print("✗ nvidia-smi timeout")
        all_checks_passed = False
    except FileNotFoundError:
        print("✗ nvidia-smi not found")
        print("  Please install NVIDIA GPU drivers")
        all_checks_passed = False

    print()

    # Check dataset
    print("Checking dataset...")
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
            print("✗ Dataset validation failed")
            all_checks_passed = False
        else:
            print("✓ Dataset structure validated")
    else:
        print("? Dataset validation skipped (utils not available)")

    print()

    return all_checks_passed

def check_disk_space():
    """Check available disk space"""
    print("Checking disk space...")
    try:
        # Check current directory disk space
        statvfs = os.statvfs('.')
        free_space_gb = (statvfs.f_frsize * statvfs.f_available) / (1024**3)
        
        print(f"Available disk space: {free_space_gb:.1f} GB")
        
        if free_space_gb < 10:
            print("✗ Warning: Low disk space (< 10 GB)")
            print("  Training may fail due to insufficient space for:")
            print("    - Model checkpoints")
            print("    - Log files")
            print("    - Temporary files")
            return False
        else:
            print("✓ Sufficient disk space available")
            return True
            
    except Exception as e:
        print(f"? Could not check disk space: {e}")
        return True  # Don't fail if we can't check

def create_gpu_directories():
    """Create necessary directories for GPU training"""
    print("Creating GPU training directories...")
    
    directories = [
        "training/checkpoints_gpu",
        "training/logs_gpu", 
        "training/output_gpu",
        "training/graphs"
    ]
    
    all_created = True
    
    for dir_path in directories:
        try:
            os.makedirs(dir_path, exist_ok=True)
            print(f"✓ Directory ready: {dir_path}")
        except Exception as e:
            print(f"✗ Failed to create {dir_path}: {e}")
            all_created = False
    
    return all_created

def run_gpu_training(config_path=None, dry_run=False):
    """Run the GPU-optimized training"""
    print("=" * 60)
    print("STARTING GPU TRAINING")
    print("=" * 60)
    
    # Change to training directory
    os.chdir("training")
    
    # Prepare command
    cmd = [sys.executable, "train_tfnet_gpu.py"]
    
    if config_path:
        cmd.extend(["--config", config_path])
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Working directory: {os.getcwd()}")
    print()
    
    if dry_run:
        print("Dry run mode - would execute the above command")
        return True
    
    try:
        # Set environment variables for better GPU performance
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
        env['MINDSPORE_HCCL_CONF_FILE'] = ''  # Clear any distributed settings
        
        # Run training
        result = subprocess.run(cmd, env=env)
        
        if result.returncode == 0:
            print("Training completed successfully!")
            return True
        else:
            print(f"Training failed with return code: {result.returncode}")
            return False
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return False
    except Exception as e:
        print(f"Training failed with error: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="GPU-Optimized TFNet Training Launcher")
    parser.add_argument("--config", type=str, default="configs/gpu_config.json",
                       help="Path to GPU configuration file")
    parser.add_argument("--skip-checks", action="store_true",
                       help="Skip environment checks")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be executed without running")
    parser.add_argument("--force", action="store_true",
                       help="Force training even if checks fail")
    
    args = parser.parse_args()
    
    print("GPU-Optimized TFNet Training Launcher")
    print("=" * 60)
    
    # Check environment unless skipped
    if not args.skip_checks:
        print("Running environment checks...")
        
        gpu_check_passed = check_gpu_environment()
        disk_check_passed = check_disk_space()
        dir_creation_passed = create_gpu_directories()
        
        all_checks_passed = gpu_check_passed and disk_check_passed and dir_creation_passed
        
        if not all_checks_passed and not args.force:
            print("\n" + "=" * 60)
            print("ENVIRONMENT CHECK FAILED")
            print("=" * 60)
            print("Some checks failed. Please fix the issues above.")
            print("Or use --force to proceed anyway (not recommended).")
            print("Or use --skip-checks to skip all checks.")
            return 1
        
        if not all_checks_passed and args.force:
            print("\n" + "=" * 60)
            print("WARNING: Proceeding despite failed checks")
            print("=" * 60)
    
    # Run training
    success = run_gpu_training(args.config, args.dry_run)
    
    if success:
        print("\n" + "=" * 60)
        print("SUCCESS: GPU training completed")
        print("=" * 60)
        return 0
    else:
        print("\n" + "=" * 60)
        print("FAILED: GPU training failed")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    exit(main())
