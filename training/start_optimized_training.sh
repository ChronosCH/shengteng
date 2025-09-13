#!/bin/bash

# GPU内存优化训练启动脚本
# 使用优化配置启动训练，避免显存不足问题

echo "🚀 启动GPU优化的手语识别训练"
echo "=================================="

# 切换到训练目录
cd "$(dirname "$0")"

# 检查CUDA环境
echo "📋 检查CUDA环境..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi未找到，请检查CUDA安装"
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

# 激活conda环境
echo "🔧 激活conda环境..."
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "当前环境: $CONDA_DEFAULT_ENV"
else
    echo "⚠️  建议激活mindspore-gpu环境: conda activate mindspore-gpu"
fi

# 运行内存优化脚本
echo "🧹 运行内存优化..."
python gpu_memory_optimizer.py
if [ $? -ne 0 ]; then
    echo "⚠️  内存优化检测到问题，但继续训练..."
fi

# 设置优化的环境变量
export CUDA_LAUNCH_BLOCKING=1
export CUDA_CACHE_DISABLE=1
export MS_DEV_ENABLE_FALLBACK=0

echo "🎯 启动优化的训练..."
echo "配置文件: configs/gpu_config.json"
echo "优化设置:"
echo "  - Batch size: 1"
echo "  - Hidden size: 256" 
echo "  - Crop size: 160x160"
echo "  - Max frames: 100"
echo "  - Max device memory: 4GB"

# 启动训练
python train_tfnet_gpu.py --config configs/gpu_config.json

echo "训练完成!"
