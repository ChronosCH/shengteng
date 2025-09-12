#!/bin/bash
"""
GPU训练环境一键设置和测试脚本
One-click setup and test script for GPU training environment
"""

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# 日志函数
log() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 显示标题
show_banner() {
    echo -e "${PURPLE}"
    echo "=========================================="
    echo "    GPU训练环境一键设置脚本"
    echo "    GPU Training Setup Script" 
    echo "=========================================="
    echo -e "${NC}"
}

# 检查根目录
check_root_directory() {
    log "检查项目根目录..."
    
    if [[ ! -d "training" ]]; then
        error "请在项目根目录运行此脚本！"
        error "当前目录: $(pwd)"
        error "预期找到: training/ 文件夹"
        exit 1
    fi
    
    success "项目根目录检查通过"
}

# 检查并激活conda环境
setup_conda_environment() {
    log "检查conda环境..."
    
    if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
        warning "未检测到激活的conda环境"
        
        if command -v conda >/dev/null 2>&1; then
            log "尝试激活mindspore-gpu环境..."
            source $(conda info --base)/etc/profile.d/conda.sh
            conda activate mindspore-gpu 2>/dev/null || {
                error "无法激活mindspore-gpu环境"
                error "请先创建并安装MindSpore GPU环境:"
                echo "  conda create -n mindspore-gpu python=3.8"
                echo "  conda activate mindspore-gpu"
                echo "  pip install mindspore-gpu"
                exit 1
            }
        else
            error "未找到conda命令"
            exit 1
        fi
    fi
    
    if [[ "$CONDA_DEFAULT_ENV" != "mindspore-gpu" ]]; then
        warning "当前环境: $CONDA_DEFAULT_ENV"
        warning "切换到mindspore-gpu环境..."
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate mindspore-gpu || {
            error "无法激活mindspore-gpu环境"
            exit 1
        }
    fi
    
    success "Conda环境: $CONDA_DEFAULT_ENV"
}

# 安装依赖包
install_dependencies() {
    log "检查Python依赖包..."
    
    # 检查关键依赖
    dependencies=(
        "mindspore"
        "opencv-python"
        "numpy"
    )
    
    missing_deps=()
    
    for dep in "${dependencies[@]}"; do
        if ! python -c "import ${dep}" 2>/dev/null; then
            missing_deps+=("$dep")
        fi
    done
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        warning "发现缺失依赖: ${missing_deps[*]}"
        log "安装缺失的依赖包..."
        
        for dep in "${missing_deps[@]}"; do
            case $dep in
                "mindspore")
                    error "MindSpore未安装，请手动安装MindSpore GPU版本"
                    echo "  pip install mindspore-gpu"
                    exit 1
                    ;;
                "opencv-python")
                    pip install opencv-python
                    ;;
                *)
                    pip install "$dep"
                    ;;
            esac
        done
    fi
    
    success "所有依赖包已安装"
}

# 创建必要目录
create_directories() {
    log "创建GPU训练目录..."
    
    directories=(
        "training/configs"
        "training/checkpoints_gpu"
        "training/logs_gpu"
        "training/output_gpu"
        "training/graphs"
    )
    
    for dir in "${directories[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            success "创建目录: $dir"
        else
            log "目录已存在: $dir"
        fi
    done
    
    success "所有目录创建完成"
}

# 检查GPU可用性
test_gpu_availability() {
    log "测试GPU可用性..."
    
    # 检查nvidia-smi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        error "nvidia-smi未找到，请安装NVIDIA驱动"
        return 1
    fi
    
    # 检查GPU
    if ! nvidia-smi >/dev/null 2>&1; then
        error "GPU不可用或驱动有问题"
        return 1
    fi
    
    # 获取GPU信息
    gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
    success "GPU检测成功: $gpu_info"
    
    # 测试MindSpore GPU
    if ! python -c "
import mindspore as ms
from mindspore import context
context.set_context(device_target='GPU')
test = ms.Tensor([[1.0]], ms.float32)
result = test + 1
print('MindSpore GPU测试成功')
" 2>/dev/null; then
        error "MindSpore GPU功能测试失败"
        return 1
    fi
    
    success "MindSpore GPU功能测试通过"
    return 0
}

# 设置权限
set_permissions() {
    log "设置脚本执行权限..."
    
    scripts=(
        "training/start_training_gpu.sh"
        "training/train_tfnet_gpu.py"
        "training/start_training_gpu.py"
        "training/quick_start_gpu.py"
        "training/test_gpu_setup.py"
    )
    
    for script in "${scripts[@]}"; do
        if [[ -f "$script" ]]; then
            chmod +x "$script"
            success "设置权限: $script"
        else
            warning "脚本不存在: $script"
        fi
    done
}

# 运行系统测试
run_system_test() {
    log "运行GPU系统测试..."
    
    cd training
    
    if python test_gpu_setup.py; then
        success "GPU系统测试全部通过！"
        return 0
    else
        warning "GPU系统测试有部分失败"
        return 1
    fi
}

# 显示下一步提示
show_next_steps() {
    echo
    echo -e "${GREEN}=========================================="
    echo -e "          设置完成！"
    echo -e "==========================================${NC}"
    echo
    echo "🚀 现在您可以开始GPU训练："
    echo
    echo "1. 快速启动（推荐）："
    echo "   cd training && python quick_start_gpu.py"
    echo
    echo "2. 使用Shell脚本："
    echo "   cd training && ./start_training_gpu.sh"
    echo
    echo "3. 直接运行训练："
    echo "   cd training && python train_tfnet_gpu.py"
    echo
    echo "📚 文档："
    echo "   - GPU训练指南: training/README_GPU_TRAINING.md"
    echo "   - 配置文件: training/configs/gpu_config.json"
    echo
    echo "🔧 如需调试："
    echo "   cd training && python test_gpu_setup.py"
    echo
    echo "📊 监控训练："
    echo "   - GPU使用: watch -n 1 nvidia-smi"
    echo "   - 训练日志: tail -f training/logs_gpu/*.log"
    echo
}

# 主函数
main() {
    show_banner
    
    # 检查基础环境
    check_root_directory
    setup_conda_environment
    install_dependencies
    create_directories
    set_permissions
    
    # 测试GPU
    if test_gpu_availability; then
        success "GPU环境设置成功！"
    else
        error "GPU环境设置失败，请检查GPU驱动和MindSpore安装"
        exit 1
    fi
    
    # 运行系统测试
    if run_system_test; then
        success "所有测试通过，GPU训练环境就绪！"
    else
        warning "部分测试失败，但基础功能可用"
    fi
    
    show_next_steps
}

# 执行主函数
main "$@"
