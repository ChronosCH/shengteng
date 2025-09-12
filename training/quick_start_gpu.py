#!/usr/bin/env python3
"""
Quick GPU Training Launcher
Simple script to quickly start GPU training with proper environment checks
"""

import os
import sys
import subprocess
import time

def check_environment():
    """Quick environment check"""
    print("=" * 50)
    print("QUICK ENVIRONMENT CHECK")
    print("=" * 50)
    
    # Check conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
    print(f"Conda Environment: {conda_env}")
    
    if 'mindspore-gpu' not in conda_env:
        print("❌ Wrong conda environment!")
        print("🔧 Please run: conda activate mindspore-gpu")
        return False
    else:
        print("✅ Correct conda environment")
    
    # Check MindSpore
    try:
        import mindspore as ms
        print(f"✅ MindSpore {ms.__version__} imported successfully")
    except ImportError:
        print("❌ MindSpore not found!")
        return False
    
    # Quick GPU test
    try:
        from mindspore import context
        context.set_context(device_target="GPU")
        test_tensor = ms.Tensor([[1.0, 2.0]], ms.float32)
        result = test_tensor + 1
        print("✅ GPU is accessible")
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False
    
    # Check files
    required_files = [
        "training/train_tfnet_gpu.py",
        "training/configs/gpu_config.json"
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ Found: {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
            return False
    
    return True

def create_directories():
    """Create necessary directories"""
    dirs = [
        "training/checkpoints_gpu",
        "training/logs_gpu", 
        "training/output_gpu"
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"✅ Directory ready: {d}")

def main():
    print("🚀 Quick GPU Training Launcher")
    
    if not check_environment():
        print("\n❌ Environment check failed!")
        print("Please fix the issues above and try again.")
        return 1
    
    print("\n🔧 Setting up directories...")
    create_directories()
    
    print("\n🏃 Starting GPU training...")
    print("=" * 50)
    
    try:
        # Change to training directory and run
        os.chdir("training")
        
        cmd = [sys.executable, "train_tfnet_gpu.py", "--config", "configs/gpu_config.json"]
        
        print(f"Executing: {' '.join(cmd)}")
        print(f"Working directory: {os.getcwd()}")
        print()
        
        # Set GPU environment variables
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0'
        
        # Run training
        process = subprocess.Popen(
            cmd, 
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        rc = process.poll()
        
        if rc == 0:
            print("\n🎉 Training completed successfully!")
            return 0
        else:
            print(f"\n❌ Training failed with return code: {rc}")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
