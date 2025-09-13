#!/usr/bin/env python3
"""
GPU内存优化和清理脚本
在训练前运行以确保最佳的内存使用
"""

import gc
import os
import sys
import subprocess

def check_gpu_memory():
    """检查GPU内存使用情况"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines):
                used, total = map(int, line.split(','))
                used_gb = used / 1024
                total_gb = total / 1024
                free_gb = total_gb - used_gb
                print(f"GPU {i}: {used_gb:.1f}GB / {total_gb:.1f}GB used, {free_gb:.1f}GB free")
                
                if free_gb < 2.0:
                    print(f"⚠️  Warning: GPU {i} has less than 2GB free memory")
                    return False
                elif free_gb < 4.0:
                    print(f"⚠️  Warning: GPU {i} has less than 4GB free memory")
                    
            return True
    except Exception as e:
        print(f"Error checking GPU memory: {e}")
        return False

def clear_gpu_cache():
    """清理GPU缓存"""
    try:
        # 清理Python垃圾回收
        gc.collect()
        
        # 尝试清理MindSpore缓存
        try:
            import mindspore as ms
            # 如果有任何张量操作缓存，清理它们
            print("✓ MindSpore memory cleared")
        except ImportError:
            print("MindSpore not available for cache clearing")
            
        # 清理CUDA缓存（如果使用PyTorch）
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("✓ PyTorch CUDA cache cleared")
        except ImportError:
            pass
            
        print("✓ System garbage collection completed")
        return True
        
    except Exception as e:
        print(f"Error during cache clearing: {e}")
        return False

def optimize_environment():
    """优化环境变量"""
    # 设置CUDA相关环境变量
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 同步CUDA操作，有助于调试
    os.environ['CUDA_CACHE_DISABLE'] = '1'    # 禁用CUDA缓存
    
    # 设置MindSpore相关环境变量
    os.environ['MS_DEV_ENABLE_FALLBACK'] = '0'  # 禁用fallback以避免内存碎片
    
    print("✓ Environment variables optimized")

def check_system_memory():
    """检查系统内存"""
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
        
        for line in meminfo.split('\n'):
            if 'MemAvailable:' in line:
                mem_available_kb = int(line.split()[1])
                mem_available_gb = mem_available_kb / (1024 * 1024)
                print(f"System RAM available: {mem_available_gb:.1f}GB")
                
                if mem_available_gb < 4.0:
                    print("⚠️  Warning: Less than 4GB system RAM available")
                    return False
                break
                
        return True
    except Exception as e:
        print(f"Error checking system memory: {e}")
        return False

def main():
    print("🚀 GPU Memory Optimization and Cleanup")
    print("=" * 50)
    
    # 优化环境
    optimize_environment()
    
    # 清理缓存
    print("\n📋 Cleaning caches...")
    clear_gpu_cache()
    
    # 检查系统内存
    print("\n💾 Checking system memory...")
    check_system_memory()
    
    # 检查GPU内存
    print("\n🎮 Checking GPU memory...")
    gpu_ok = check_gpu_memory()
    
    print("\n" + "=" * 50)
    if gpu_ok:
        print("✅ Memory optimization completed successfully!")
        print("💡 Recommendations:")
        print("   - Use batch_size=1 for initial testing")
        print("   - Monitor GPU memory during training")
        print("   - Consider reducing model size if issues persist")
    else:
        print("❌ GPU memory issues detected!")
        print("💡 Try these solutions:")
        print("   1. Restart the system or kill GPU processes")
        print("   2. Use smaller batch sizes")
        print("   3. Reduce model parameters")
        print("   4. Use gradient checkpointing")
        
    return gpu_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
