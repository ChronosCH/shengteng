#!/bin/bash
# 测试cuBLAS修复的启动脚本

echo "=== cuBLAS错误修复测试 ==="
echo "确保已激活环境: conda activate mindspore_gpu_env"
echo

# 检查环境
if [[ "$CONDA_DEFAULT_ENV" != "mindspore_gpu_env" ]]; then
    echo "警告: 当前环境不是 mindspore_gpu_env"
    echo "请运行: conda activate mindspore_gpu_env"
    echo
fi

# 设置CUDA环境变量以优化cuBLAS
export CUDA_LAUNCH_BLOCKING=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "设置的环境变量:"
echo "CUDA_LAUNCH_BLOCKING=1 (用于调试)"
echo "CUBLAS_WORKSPACE_CONFIG=:4096:8 (确定性行为)"
echo

# 1. 首先测试基本的cuBLAS修复
echo "1. 测试cuBLAS修复模块..."
python cublas_fixes.py
echo

# 2. 测试修复后的模型
echo "2. 测试修复后的TFNet模型..."
python test_fixed_model.py
echo

# 3. 如果基础测试通过，运行完整训练
if [ $? -eq 0 ]; then
    echo "3. 基础测试通过，开始训练测试..."
    echo "运行命令: python train_tfnet_gpu.py --config configs/safe_gpu_config.json"
    echo
    
    # 询问是否继续
    read -p "是否继续运行完整训练? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python train_tfnet_gpu.py --config configs/safe_gpu_config.json
    else
        echo "跳过完整训练测试"
    fi
else
    echo "基础测试失败，请检查错误信息"
fi

echo
echo "=== 测试完成 ==="
