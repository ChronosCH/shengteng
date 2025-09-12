#!/bin/bash
# CUDA 环境配置脚本
# 用于配置 MindSpore GPU 版本所需的环境变量

echo "配置 CUDA 环境变量..."

# 检测 CUDA 安装路径
CUDA_PATH="/usr/local/cuda-11.6"

if [ -d "$CUDA_PATH" ]; then
    echo "找到 CUDA 安装路径: $CUDA_PATH"
    
    # 设置环境变量
    export CUDA_HOME="$CUDA_PATH"
    export PATH="$CUDA_PATH/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"
    
    echo "环境变量已设置:"
    echo "  CUDA_HOME=$CUDA_HOME"
    echo "  PATH=$PATH"
    echo "  LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    
    # 添加到 bashrc 以实现永久配置
    BASHRC_FILE="$HOME/.bashrc"
    
    # 检查是否已经配置
    if ! grep -q "CUDA_HOME.*cuda-11.6" "$BASHRC_FILE" 2>/dev/null; then
        echo ""
        echo "将环境变量添加到 ~/.bashrc 以实现永久配置..."
        
        cat >> "$BASHRC_FILE" << EOF

# CUDA 环境配置 (MindSpore GPU)
export CUDA_HOME=/usr/local/cuda-11.6
export PATH=/usr/local/cuda-11.6/bin:\$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:\$LD_LIBRARY_PATH
EOF
        
        echo "✓ 环境变量已添加到 ~/.bashrc"
        echo "  请运行 'source ~/.bashrc' 或重新登录以使配置生效"
    else
        echo "✓ 环境变量已存在于 ~/.bashrc 中"
    fi
    
else
    echo "❌ 未找到 CUDA 安装路径: $CUDA_PATH"
    echo "请确保 CUDA 已正确安装"
    exit 1
fi

echo ""
echo "验证 CUDA 配置:"
if command -v nvcc &> /dev/null; then
    echo "✓ nvcc 可用: $(nvcc --version | grep -o 'release [0-9]*\.[0-9]*')"
else
    echo "❌ nvcc 不可用"
fi

if command -v nvidia-smi &> /dev/null; then
    echo "✓ nvidia-smi 可用"
else
    echo "❌ nvidia-smi 不可用"
fi

echo ""
echo "配置完成！现在您可以:"
echo "1. 运行 MindSpore GPU 版本"
echo "2. 使用提供的验证脚本测试环境"
echo "3. 开始您的深度学习项目"
