#!/bin/bash

# GPU优化的TFNet训练启动脚本
# 自动激活conda环境并启动训练

set -e  # 遇到错误时退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo "============================================================"
    echo "TFNet GPU Training System with Environment Management"
    echo "Continuous Sign Language Recognition"
    echo "============================================================"
}

# 检查conda是否安装
check_conda() {
    if ! command -v conda &> /dev/null; then
        print_error "Conda is not installed or not in PATH"
        print_error "Please install Anaconda or Miniconda first"
        exit 1
    fi
    print_success "Conda found: $(conda --version)"
}

# 检查环境是否存在
check_environment() {
    local env_name="$1"
    if conda env list | grep -q "^${env_name} "; then
        print_success "Environment '${env_name}' exists"
        return 0
    else
        print_error "Environment '${env_name}' does not exist"
        print_info "Please create it first or check the environment name"
        return 1
    fi
}

# 激活环境并运行训练
run_training() {
    local config_file="${1:-configs/gpu_config.json}"
    
    print_header
    
    # 检查conda
    check_conda
    
    # 检查mind环境
    if ! check_environment "mind"; then
        print_error "Failed to find 'mind' environment"
        print_info "Please run: conda create -n mind python=3.8"
        exit 1
    fi
    
    print_info "Activating conda environment 'mind'..."
    
    # 初始化conda (确保conda命令在脚本中可用)
    eval "$(conda shell.bash hook)"
    
    # 激活环境
    conda activate mind
    
    if [[ "$CONDA_DEFAULT_ENV" != "mind" ]]; then
        print_error "Failed to activate 'mind' environment"
        exit 1
    fi
    
    print_success "Environment activated: $CONDA_DEFAULT_ENV"
    
    # 检查Python和MindSpore
    print_info "Checking Python and MindSpore..."
    python -c "import sys; print(f'Python version: {sys.version}')"
    
    if python -c "import mindspore as ms; print(f'MindSpore version: {ms.__version__}')" 2>/dev/null; then
        print_success "MindSpore is available"
    else
        print_error "MindSpore not found in the environment"
        print_info "Please install MindSpore: pip install mindspore-gpu"
        exit 1
    fi
    
    # 检查GPU
    if nvidia-smi &> /dev/null; then
        print_success "NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    else
        print_warning "No NVIDIA GPU detected or nvidia-smi not available"
    fi
    
    # 检查配置文件
    if [[ ! -f "$config_file" ]]; then
        print_error "Configuration file not found: $config_file"
        exit 1
    fi
    print_success "Configuration file found: $config_file"
    
    # 启动训练
    print_info "Starting TFNet GPU training..."
    print_info "Command: python train_tfnet_gpu.py --config $config_file"
    print_info "Press Ctrl+C to stop training"
    
    echo "============================================================"
    
    # 运行训练脚本
    python train_tfnet_gpu.py --config "$config_file"
    
    local exit_code=$?
    
    echo "============================================================"
    
    if [[ $exit_code -eq 0 ]]; then
        print_success "Training completed successfully!"
    else
        print_error "Training failed with exit code: $exit_code"
    fi
    
    return $exit_code
}

# 显示使用帮助
show_help() {
    echo "Usage: $0 [CONFIG_FILE]"
    echo ""
    echo "Arguments:"
    echo "  CONFIG_FILE    Path to configuration file (default: configs/gpu_config.json)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Use default config"
    echo "  $0 configs/gpu_config.json          # Use specific config"
    echo ""
    echo "Environment Requirements:"
    echo "  - Conda environment named 'mind' must exist"
    echo "  - MindSpore must be installed in the 'mind' environment"
    echo "  - NVIDIA GPU and drivers (optional, for GPU training)"
    echo ""
}

# 主函数
main() {
    # 解析命令行参数
    case "${1:-}" in
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            run_training "$1"
            ;;
    esac
}

# 捕获中断信号
trap 'print_warning "Training interrupted by user"; exit 130' INT TERM

# 运行主函数
main "$@"
