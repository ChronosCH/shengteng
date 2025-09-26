#!/bin/bash
# ==============================================================================
# ASL项目完整环境安装脚本
# 从安装miniconda开始，到创建conda环境，安装所有依赖包
# 包含MindSpore GPU安装（使用特定的安装方法）
# ==============================================================================

set -euo pipefail

# --- 配置参数 ---
ENV_NAME="asl_env"
PYTHON_VERSION="3.9"
CUDA_TOOLKIT_VER="10.1"
CUDNN_VER="7.6.5"
CONDA_CHANNEL="conda-forge"

# MindSpore相关配置
MS_WHL_URL="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.2.0/MindSpore/unified/x86_64/mindspore-2.2.0-cp39-cp39-linux_x86_64.whl"
MS_TRUSTED_HOST="ms-release.obs.cn-north-4.myhuaweicloud.com"

# Miniconda下载URL
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
MINICONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
CONDA_INSTALL_DIR="$HOME/miniconda3"

echo "=== ASL项目环境完整安装脚本 ==="
echo "本脚本将："
echo "1. 安装Miniconda（如果尚未安装）"
echo "2. 创建Python 3.9 conda环境"
echo "3. 安装CUDA工具包和cuDNN"
echo "4. 安装MindSpore GPU版本"
echo "5. 安装项目所需的其他依赖包"
echo ""
echo "按Enter继续，Ctrl+C取消..."
read -r

# ==============================================================================
# 第1步：检查并安装Miniconda
# ==============================================================================
echo "=== 第1步：检查并安装Miniconda ==="

if command -v conda >/dev/null 2>&1; then
    echo "✅ 检测到conda已安装："
    conda --version
else
    echo "📦 conda未安装，开始安装Miniconda..."
    
    # 下载Miniconda安装包
    if [ ! -f "$MINICONDA_INSTALLER" ]; then
        echo "下载Miniconda安装包..."
        wget -q --show-progress "$MINICONDA_URL" -O "$MINICONDA_INSTALLER"
    else
        echo "Miniconda安装包已存在，跳过下载"
    fi
    
    # 安装Miniconda
    echo "安装Miniconda到 $CONDA_INSTALL_DIR..."
    bash "$MINICONDA_INSTALLER" -b -p "$CONDA_INSTALL_DIR"
    
    # 初始化conda
    echo "初始化conda..."
    "$CONDA_INSTALL_DIR/bin/conda" init bash
    
    # 添加conda到PATH
    export PATH="$CONDA_INSTALL_DIR/bin:$PATH"
    
    echo "✅ Miniconda安装完成"
    echo "请重新启动终端或运行 'source ~/.bashrc' 来激活conda"
    echo "然后重新运行本脚本"
    exit 0
fi

# ==============================================================================
# 第2步：GPU驱动检查
# ==============================================================================
echo "=== 第2步：GPU驱动检查 ==="

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "✅ NVIDIA 驱动信息："
    nvidia-smi
else
    echo "⚠️  警告：未检测到 nvidia-smi。请确认已安装并加载 NVIDIA 驱动。"
    echo "如果您的系统没有GPU，某些功能可能无法使用。"
fi

# ==============================================================================
# 第3步：设置conda环境
# ==============================================================================
echo "=== 第3步：设置conda环境 ==="

# 让后续shell能正确激活conda
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# 创建环境（若不存在才创建；存在则跳过）
if conda info --envs | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    echo "✅ Conda环境 '$ENV_NAME' 已存在，跳过创建"
else
    echo "📦 创建Conda环境：$ENV_NAME (Python ${PYTHON_VERSION})"
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" 
    echo "✅ 环境创建完成"
fi

# ==============================================================================
# 第4步：环境验证和Python版本检查
# ==============================================================================
echo "=== 第4步：环境验证和Python版本检查 ==="

# 环境路径检查
ENV_PREFIX="$(conda run -n "$ENV_NAME" bash -lc 'printf %s "$CONDA_PREFIX"' 2>/dev/null || true)"
if [ -z "${ENV_PREFIX:-}" ]; then
    ENV_PREFIX="$(bash -lc "source '${CONDA_BASE}/etc/profile.d/conda.sh' && conda activate '${ENV_NAME}' && printf %s \"\$CONDA_PREFIX\"" 2>/dev/null || true)"
fi
if [ -z "${ENV_PREFIX:-}" ] || [ ! -d "$ENV_PREFIX" ]; then
    echo "❌ 错误：无法获取环境前缀，请确认环境 '$ENV_NAME' 存在且可用"
    exit 1
fi

# Python版本检查
ENV_PY_VER="$(conda run -n "$ENV_NAME" python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))' 2>/dev/null | head -n1 | tr -d '\r' || true)"
if [ -z "${ENV_PY_VER:-}" ]; then
    ENV_PY_VER="$(bash -lc "source '${CONDA_BASE}/etc/profile.d/conda.sh' && conda activate '${ENV_NAME}' && python -c 'import sys; print(\".\".join(map(str, sys.version_info[:2])))'" 2>/dev/null | head -n1 | tr -d '\r' || true)"
fi

if [ -z "${ENV_PY_VER:-}" ]; then
    echo "❌ 错误：无法检测到环境 '$ENV_NAME' 的Python版本"
    exit 1
fi

if [ "$ENV_PY_VER" != "$PYTHON_VERSION" ]; then
    echo "❌ 错误：环境 '$ENV_NAME' 的Python版本为 ${ENV_PY_VER}，但需要Python 3.9"
    echo "请删除现有环境后重新运行脚本：conda env remove -n $ENV_NAME"
    exit 1
fi

echo "✅ 环境验证完成"
echo "   环境名称：$ENV_NAME"
echo "   Python版本：$ENV_PY_VER"
echo "   环境路径：$ENV_PREFIX"

# ==============================================================================
# 第5步：安装CUDA工具包和cuDNN
# ==============================================================================
echo "=== 第5步：安装CUDA工具包和cuDNN ==="

echo "📦 安装cudatoolkit=${CUDA_TOOLKIT_VER} 和 cudnn=${CUDNN_VER}..."
conda install -n "$ENV_NAME" -c "$CONDA_CHANNEL" cudatoolkit="$CUDA_TOOLKIT_VER" cudnn="$CUDNN_VER" -y

echo "✅ CUDA工具包和cuDNN安装完成"

# ==============================================================================
# 第6步：配置CUDA运行时环境
# ==============================================================================
echo "=== 第6步：配置CUDA运行时环境 ==="

# 为避免 /usr/bin/bash 受 conda 的 libtinfo 影响出现
# "/usr/bin/bash: .../libtinfo.so.6: no version information available" 的提示，
# 这里不再在激活钩子中修改 LD_LIBRARY_PATH。
# 对于通过 conda 安装的 cudatoolkit/cudnn，通常无需设置 LD_LIBRARY_PATH，
# 其 rpath 已能正确解析依赖。如仍需自定义，可按需手动导出：
#   export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
# 或仅在运行 Python/训练脚本前临时设置。

echo "ℹ️ 跳过在激活脚本中修改 LD_LIBRARY_PATH（避免与系统 bash 的 libtinfo 冲突）"

# ==============================================================================
# 第7步：安装MindSpore
# ==============================================================================
echo "=== 第7步：安装MindSpore GPU版本 ==="

echo "📦 安装MindSpore 2.2.0 (unified, cp39)..."
conda run -n "$ENV_NAME" python -m pip install --no-cache-dir "$MS_WHL_URL" --trusted-host "$MS_TRUSTED_HOST"

echo "✅ MindSpore安装完成"

# ==============================================================================
# 第8步：安装项目依赖包
# ==============================================================================
echo "=== 第8步：安装项目依赖包 ==="

# 检查requirements.txt是否存在
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"

if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "📦 从requirements.txt安装项目依赖..."
    conda run -n "$ENV_NAME" python -m pip install -r "$REQUIREMENTS_FILE"
    echo "✅ 项目依赖安装完成"
else
    echo "⚠️  未找到requirements.txt文件，安装常用的机器学习包..."
    conda run -n "$ENV_NAME" python -m pip install \
        numpy \
        pandas \
        matplotlib \
        seaborn \
        scikit-learn \
        opencv-python \
        tqdm \
        jupyter \
        ipykernel
    echo "✅ 常用包安装完成"
fi

# 安装额外的数据科学包
echo "📦 安装conda包管理的数据科学工具..."
conda install -n "$ENV_NAME" -c conda-forge \
    ffmpeg \
    pillow \
    scipy \
    -y

echo "✅ 额外依赖包安装完成"

# ==============================================================================
# 第9步：验证安装
# ==============================================================================
echo "=== 第9步：验证安装 ==="

echo "🔍 检查已安装的CUDA/cuDNN包："
conda run -n "$ENV_NAME" conda list | grep -E 'cudatoolkit|cudnn' || true

echo ""
echo "🔍 运行MindSpore GPU验证..."
conda run -n "$ENV_NAME" bash -lc '
python - <<PY
import os
import sys
print("=== 环境验证 ===")
print("Python版本:", sys.version.split()[0])
print("Python路径:", sys.executable)

# 检查LD_LIBRARY_PATH
ld_path = os.environ.get("LD_LIBRARY_PATH", "")
print("LD_LIBRARY_PATH前120字符:", (ld_path[:120] + "...") if ld_path else "(空)")

# 检查基础包
try:
    import numpy as np
    print("✅ NumPy版本:", np.__version__)
except ImportError as e:
    print("❌ NumPy导入失败:", e)

try:
    import cv2
    print("✅ OpenCV版本:", cv2.__version__)
except ImportError as e:
    print("❌ OpenCV导入失败:", e)

# 检查MindSpore
try:
    import mindspore as ms
    from mindspore import Tensor, ops
    print("✅ MindSpore版本:", ms.__version__)
    
    # 尝试GPU上下文
    try:
        ms.set_context(device_target="GPU")
        x = Tensor(np.ones((2,3), dtype=np.float32))
        y = Tensor(np.ones((2,3), dtype=np.float32))
        out = ops.add(x, y)
        print("✅ GPU加法测试成功，结果和=", float(out.asnumpy().sum()))
        print("✅ 当前device_target =", ms.get_context("device_target"))
    except Exception as gpu_e:
        print("⚠️  GPU上下文初始化失败:", repr(gpu_e))
        print("   尝试CPU模式...")
        try:
            ms.set_context(device_target="CPU")
            x = Tensor(np.ones((2,3), dtype=np.float32))
            y = Tensor(np.ones((2,3), dtype=np.float32))
            out = ops.add(x, y)
            print("✅ CPU加法测试成功，结果和=", float(out.asnumpy().sum()))
            print("✅ 当前device_target =", ms.get_context("device_target"))
        except Exception as cpu_e:
            print("❌ CPU模式也失败:", repr(cpu_e))
            
except ImportError as e:
    print("❌ MindSpore导入失败:", e)

print("=== 验证完成 ===")
PY
'

# ==============================================================================
# 第10步：创建Jupyter kernel
# ==============================================================================
echo "=== 第10步：创建Jupyter kernel ==="

echo "📝 为环境创建Jupyter kernel..."
conda run -n "$ENV_NAME" python -m ipykernel install --user --name "$ENV_NAME" --display-name "ASL Environment ($ENV_NAME)"

echo "✅ Jupyter kernel创建完成"

# ==============================================================================
# 安装完成总结
# ==============================================================================
echo ""
echo "=========================================="
echo "            🎉 安装完成！ 🎉"
echo "=========================================="
echo ""
echo "📋 安装总结："
echo "   ✅ Miniconda: $(conda --version 2>/dev/null || echo '已安装')"
echo "   ✅ 环境名称: $ENV_NAME"
echo "   ✅ Python版本: $PYTHON_VERSION"
echo "   ✅ CUDA工具包: $CUDA_TOOLKIT_VER"
echo "   ✅ cuDNN: $CUDNN_VER"
echo "   ✅ MindSpore: GPU版本 2.2.0"
echo "   ✅ 项目依赖: 已安装"
echo "   ✅ Jupyter kernel: 已创建"
echo ""
echo "🚀 使用方法："
echo "   1. 激活环境: conda activate $ENV_NAME"
echo "   2. 测试MindSpore: python -c \"import mindspore as ms; ms.set_context(device_target='GPU'); print('MindSpore版本:', ms.__version__)\""
echo "   3. 启动Jupyter: jupyter lab 或 jupyter notebook"
echo "   4. 运行训练: python main.py"
echo ""
echo "📁 项目目录: $SCRIPT_DIR"
echo ""
echo "💡 提示："
echo "   - 如需编译CUDA代码，请另行安装系统级CUDA开发工具包"
echo "   - 首次运行GPU代码时可能需要较长初始化时间"
echo "   - 如遇到GPU问题，可尝试CPU模式：ms.set_context(device_target='CPU')"
echo ""
echo "=========================================="
echo "           环境配置完成，开始编程吧！ 🚀"
echo "=========================================="