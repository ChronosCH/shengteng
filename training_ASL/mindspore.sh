#!/bin/bash
# ==============================================================================
# MindSpore GPU 安装脚本（Conda: cudatoolkit=10.1 + cudnn=7.6.5 + unified whl）
# - 不修改系统 CUDA，不需要 sudo
# - 如果 Conda 环境已存在：跳过创建（不删除、不重建）
# - 要求环境 Python 3.9（cp39 轮子），采用稳健检测（conda run + 激活兜底）
# ==============================================================================

set -euo pipefail

# --- 可配置项 ---
ENV_NAME="mind"
PYTHON_VERSION="3.9"
CUDA_TOOLKIT_VER="10.1"
CUDNN_VER="7.6.5"
CONDA_CHANNEL="conda-forge"  # 老版本包在 conda-forge 可用性更高

MS_WHL_URL="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.2.0/MindSpore/unified/x86_64/mindspore-2.2.0-cp39-cp39-linux_x86_64.whl"
MS_TRUSTED_HOST="ms-release.obs.cn-north-4.myhuaweicloud.com"

echo "=== MindSpore GPU 安装（Conda CUDA ${CUDA_TOOLKIT_VER} + cuDNN ${CUDNN_VER} + MindSpore 2.2.0 unified）==="
echo "本脚本只在 Conda 环境内安装运行时，不会改动系统级 CUDA。"
echo "按 Enter 继续，Ctrl+C 取消。"
read -r

# --- 预检查 ---
if ! command -v conda >/dev/null 2>&1; then
  echo "错误：未检测到 conda 命令。请先安装 Anaconda/Miniconda。"
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA 驱动信息（仅供参考）：" && nvidia-smi || true
else
  echo "警告：未检测到 nvidia-smi。请确认已安装并加载 NVIDIA 驱动。"
fi

# 让后续 shell 能正确激活 conda
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# --- 1) 创建环境（若不存在才创建；存在则跳过） ---
if conda info --envs | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "Conda 环境 '$ENV_NAME' 已存在：跳过创建。"
else
  echo "创建 Conda 环境：$ENV_NAME (Python ${PYTHON_VERSION}) ..."
  conda create -n "$ENV_NAME" python="$PYTHON_VERSION" 
fi

# --- 2) 环境路径 & Python 版本检查（必须为 3.9） ---
# 优先用 conda run；如果拿不到，兜底用显式激活
ENV_PREFIX="$(conda run -n "$ENV_NAME" bash -lc 'printf %s "$CONDA_PREFIX"' 2>/dev/null || true)"
if [ -z "${ENV_PREFIX:-}" ]; then
  # 兜底：显式激活取前缀
  ENV_PREFIX="$(bash -lc "source '${CONDA_BASE}/etc/profile.d/conda.sh' && conda activate '${ENV_NAME}' && printf %s \"\$CONDA_PREFIX\"" 2>/dev/null || true)"
fi
if [ -z "${ENV_PREFIX:-}" ] || [ ! -d "$ENV_PREFIX" ]; then
  echo "错误：无法获取环境前缀，请确认环境 '$ENV_NAME' 存在且可用。"
  exit 1
fi

# 稳健检测 Python 主次版本号（先 conda run，再激活兜底）
ENV_PY_VER="$(conda run -n "$ENV_NAME" python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))' 2>/dev/null | head -n1 | tr -d '\r' || true)"
if [ -z "${ENV_PY_VER:-}" ]; then
  ENV_PY_VER="$(bash -lc "source '${CONDA_BASE}/etc/profile.d/conda.sh' && conda activate '${ENV_NAME}' && python -c 'import sys; print(\".\".join(map(str, sys.version_info[:2])))'" 2>/dev/null | head -n1 | tr -d '\r' || true)"
fi

if [ -z "${ENV_PY_VER:-}" ]; then
  echo "错误：无法检测到环境 '$ENV_NAME' 的 Python 版本。请手动检查该环境是否可用。"
  exit 1
fi

if [ "$ENV_PY_VER" != "$PYTHON_VERSION" ]; then
  echo "错误：环境 '$ENV_NAME' 的 Python 版本为 ${ENV_PY_VER}，但当前安装包为 cp39（Python 3.9）。"
  echo "解决方案："
  echo "  - 新建一个 Python 3.9 的环境（例如：conda create -n ${ENV_NAME}-py39 python=3.9），"
  echo "    或修改脚本使用与你环境匹配的 MindSpore 轮子。"
  exit 1
fi

echo "已确认环境 '$ENV_NAME' 使用 Python ${ENV_PY_VER}。"
echo "环境前缀：$ENV_PREFIX"

# --- 3) 安装 cudatoolkit 与 cudnn（存在则跳过/更新） ---
echo "在环境 '$ENV_NAME' 安装/校验 cudatoolkit=${CUDA_TOOLKIT_VER} 与 cudnn=${CUDNN_VER} ..."
conda install -n "$ENV_NAME" -c "$CONDA_CHANNEL" cudatoolkit="$CUDA_TOOLKIT_VER" cudnn="$CUDNN_VER" -y

# --- 4) 写入环境激活钩子（确保动态库路径可见；重复执行会覆盖为最新内容） ---
echo "写入环境激活钩子（导出 \$CONDA_PREFIX/lib 到 LD_LIBRARY_PATH） ..."
mkdir -p "$ENV_PREFIX/etc/conda/activate.d" "$ENV_PREFIX/etc/conda/deactivate.d"

cat > "$ENV_PREFIX/etc/conda/activate.d/ms_cuda_runtime.sh" <<'EOF'
# 仅在本环境内生效：确保加载到 conda 提供的 CUDA/cuDNN 运行时动态库
if [ -d "${CONDA_PREFIX}/lib" ]; then
  if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
  else
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib"
  fi
fi
EOF

cat > "$ENV_PREFIX/etc/conda/deactivate.d/ms_cuda_runtime.sh" <<'EOF'
# 从 LD_LIBRARY_PATH 去除本环境加入的前缀
if [ -d "${CONDA_PREFIX}/lib" ] && [ -n "${LD_LIBRARY_PATH:-}" ]; then
  case "$LD_LIBRARY_PATH" in
    "${CONDA_PREFIX}/lib:"*) LD_LIBRARY_PATH="${LD_LIBRARY_PATH#${CONDA_PREFIX}/lib:}";;
  esac
  export LD_LIBRARY_PATH
fi
EOF

echo "激活钩子位置："
echo "  $ENV_PREFIX/etc/conda/activate.d/ms_cuda_runtime.sh"
echo "  $ENV_PREFIX/etc/conda/deactivate.d/ms_cuda_runtime.sh"

# --- 5) 安装 MindSpore 2.2.0（unified，cp39） ---
echo "安装 MindSpore 2.2.0（unified，cp39） ..."
conda run -n "$ENV_NAME" python -m pip install --no-cache-dir "$MS_WHL_URL" --trusted-host "$MS_TRUSTED_HOST"

# --- 6) 列表检查 ---
echo "环境内 CUDA/cuDNN 包情况："
conda run -n "$ENV_NAME" conda list | grep -E 'cudatoolkit|cudnn' || true

# --- 7) 运行验证（GPU 上下文） ---
echo "运行 MindSpore GPU 验证 ..."
conda run -n "$ENV_NAME" bash -lc '
python - <<PY
import os
import numpy as np
import mindspore as ms
from mindspore import Tensor, ops

print("Python:", os.popen("python -V").read().strip())
print("LD_LIBRARY_PATH begins with:", (os.environ.get("LD_LIBRARY_PATH","")[:120] + "...") if os.environ.get("LD_LIBRARY_PATH") else "(empty)")
print("MindSpore 版本:", ms.__version__)

try:
    ms.set_context(device_target="GPU")
    x = Tensor(np.ones((2,3), dtype=np.float32))
    y = Tensor(np.ones((2,3), dtype=np.float32))
    out = ops.add(x, y)
    print("✅ GPU 加法成功，结果和=", float(out.asnumpy().sum()))
    print("当前 device_target =", ms.get_context("device_target"))
except Exception as e:
    print("❌ MindSpore GPU 上下文初始化失败：", repr(e))
    print("排查建议：")
    print("1) 确认显卡驱动已安装（nvidia-smi 可正常输出）。")
    print("2) 保证已激活 conda 环境：conda activate", os.environ.get("CONDA_DEFAULT_ENV", "(unknown)"))
    print("3) 已安装 cudatoolkit='${CUDA_TOOLKIT_VER}' 与 cudnn='${CUDNN_VER}'（conda list 检查）。")
    print("4) 某些版本组合会提示版本不匹配，但仍可用；若依旧失败可更换 MindSpore 版本或改用官方 cuda-11.6 构建的 GPU 轮子。")
PY
'

echo ""
echo "=========================================="
echo "            安装流程结束 ✅"
echo "=========================================="
echo "使用方法："
echo "  conda activate ${ENV_NAME}"
echo "  python -c 'import mindspore as ms; ms.set_context(device_target=\"GPU\"); print(ms.__version__)'"
echo ""
echo "提示：本方案不含 nvcc（只需运行时）。如需编译器，请另装系统级 CUDA。"
echo "=========================================="
