#!/bin/bash

# TFNet集成测试运行脚本

echo "=========================================="
echo "TFNet MindSpore 集成测试"
echo "=========================================="

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: Python3 未安装"
    exit 1
fi

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "项目根目录: $PROJECT_ROOT"

# 设置Python路径
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/training:$PROJECT_ROOT/backend:$PYTHONPATH"

# 检查依赖
echo "检查依赖包..."

# 基础依赖
REQUIRED_PACKAGES=("numpy" "opencv-python" "tqdm")

for package in "${REQUIRED_PACKAGES[@]}"; do
    if python3 -c "import ${package}" 2>/dev/null; then
        echo "✓ $package 已安装"
    else
        echo "✗ $package 未安装"
        echo "请运行: pip install $package"
    fi
done

# 检查MindSpore
if python3 -c "import mindspore" 2>/dev/null; then
    MINDSPORE_VERSION=$(python3 -c "import mindspore; print(mindspore.__version__)")
    echo "✓ MindSpore $MINDSPORE_VERSION 已安装"
else
    echo "✗ MindSpore 未安装"
    echo "请运行: pip install mindspore"
    echo "或访问 https://www.mindspore.cn/install 获取安装指南"
fi

# 创建测试目录
mkdir -p temp

echo ""
echo "开始运行集成测试..."
echo "=========================================="

# 运行测试
python3 tests/test_tfnet_integration.py

TEST_EXIT_CODE=$?

echo ""
echo "=========================================="

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "🎉 所有测试通过！"
    echo "TFNet MindSpore 集成验证成功"
else
    echo "⚠ 部分测试失败"
    echo "请检查上述错误信息并解决相关问题"
fi

echo "=========================================="

exit $TEST_EXIT_CODE
