#!/usr/bin/env python3
"""
内存优化的GPU训练脚本启动器
解决MindSpore GPU训练中的内存问题
"""

import os
import sys
import gc
import subprocess

def check_gpu_memory():
    """检查GPU内存使用情况"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines):
                used, total = map(int, line.split(', '))
                print(f"GPU {i}: {used}MB / {total}MB used ({used/total*100:.1f}%)")
                if used > total * 0.8:  # If more than 80% used
                    print(f"Warning: GPU {i} memory usage is high")
        else:
            print("Could not check GPU memory")
    except FileNotFoundError:
        print("nvidia-smi not found, cannot check GPU memory")

def clear_gpu_memory():
    """清理GPU内存"""
    try:
        # Force garbage collection
        gc.collect()
        
        # Try to clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("✓ Cleared PyTorch CUDA cache")
        except ImportError:
            pass
            
        print("✓ Cleared system memory")
    except Exception as e:
        print(f"Warning: Could not clear GPU memory: {e}")

def set_memory_environment():
    """设置内存相关的环境变量"""
    # 设置MindSpore环境变量
    os.environ['MS_MEMORY_POOL_RECYCLE'] = '1'  # 启用内存池回收
    os.environ['GLOG_v'] = '2'  # 减少日志输出
    os.environ['MS_SUBMODULE_LOG_v'] = 'WARNING'  # 减少子模块日志
    
    # 设置OpenCV线程数以减少内存使用
    os.environ['OPENCV_NUM_THREADS'] = '2'
    
    # 设置MKL线程数
    os.environ['MKL_NUM_THREADS'] = '2'
    os.environ['NUMEXPR_NUM_THREADS'] = '2'
    os.environ['OMP_NUM_THREADS'] = '2'
    
    print("✓ Set memory optimization environment variables")

def main():
    """主函数"""
    print("="*60)
    print("内存优化的GPU训练启动器")
    print("="*60)
    
    # 设置环境变量
    set_memory_environment()
    
    # 检查GPU内存
    print("\nGPU内存状态:")
    check_gpu_memory()
    
    # 清理内存
    print("\n清理内存...")
    clear_gpu_memory()
    
    print("\n启动训练...")
    print("="*60)
    
    # 构建训练命令
    training_script = "train_tfnet_gpu.py"
    config_file = "configs/gpu_config.json"
    
    cmd = [
        sys.executable,  # Python可执行文件
        training_script,
        "--config", config_file
    ]
    
    try:
        # 启动训练进程
        process = subprocess.Popen(
            cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            env=os.environ.copy()
        )
        
        # 等待训练完成
        return_code = process.wait()
        
        if return_code == 0:
            print("\n训练成功完成!")
        else:
            print(f"\n训练失败，返回码: {return_code}")
            
        return return_code
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        try:
            process.terminate()
        except:
            pass
        return 1
    except Exception as e:
        print(f"\n启动训练时发生错误: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
