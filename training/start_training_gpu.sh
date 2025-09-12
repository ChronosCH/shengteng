#!/bin/bash
"""
GPU-Optimized TFNet Training Launcher Script
Ensures proper environment setup and launches GPU training
"""

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check conda environment
check_conda_env() {
    log "Checking conda environment..."
    
    if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
        error "No conda environment activated"
        error "Please run: conda activate mindspore-gpu"
        return 1
    fi
    
    if [[ "$CONDA_DEFAULT_ENV" != "mindspore-gpu" ]]; then
        warning "Current environment: $CONDA_DEFAULT_ENV"
        warning "Expected environment: mindspore-gpu"
        warning "Please run: conda activate mindspore-gpu"
        return 1
    fi
    
    success "Conda environment OK: $CONDA_DEFAULT_ENV"
    return 0
}

# Function to check GPU availability
check_gpu() {
    log "Checking GPU availability..."
    
    if ! command_exists nvidia-smi; then
        error "nvidia-smi not found. Please install NVIDIA drivers."
        return 1
    fi
    
    # Check if nvidia-smi runs successfully
    if ! nvidia-smi >/dev/null 2>&1; then
        error "nvidia-smi failed to run"
        return 1
    fi
    
    # Get GPU info
    gpu_count=$(nvidia-smi --list-gpus | wc -l)
    if [[ $gpu_count -eq 0 ]]; then
        error "No GPUs detected"
        return 1
    fi
    
    success "Found $gpu_count GPU(s)"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits | head -1
    return 0
}

# Function to check MindSpore installation
check_mindspore() {
    log "Checking MindSpore installation..."
    
    if ! python -c "import mindspore" 2>/dev/null; then
        error "MindSpore not installed or not accessible"
        error "Please install MindSpore GPU version"
        return 1
    fi
    
    # Get MindSpore version
    mindspore_version=$(python -c "import mindspore; print(mindspore.__version__)" 2>/dev/null)
    success "MindSpore version: $mindspore_version"
    
    # Test GPU functionality
    if ! python -c "
import mindspore as ms
from mindspore import context
context.set_context(device_target='GPU')
test_tensor = ms.Tensor([[1.0, 2.0], [3.0, 4.0]], ms.float32)
result = test_tensor + 1
print('GPU test successful')
" 2>/dev/null; then
        error "MindSpore GPU functionality test failed"
        return 1
    fi
    
    success "MindSpore GPU functionality OK"
    return 0
}

# Function to check disk space
check_disk_space() {
    log "Checking disk space..."
    
    # Get available space in GB
    available_space=$(df . | tail -1 | awk '{print int($4/1024/1024)}')
    
    if [[ $available_space -lt 10 ]]; then
        warning "Low disk space: ${available_space}GB available"
        warning "Training may fail due to insufficient space"
        return 1
    fi
    
    success "Sufficient disk space: ${available_space}GB available"
    return 0
}

# Function to check project structure
check_project_structure() {
    log "Checking project structure..."
    
    required_files=(
        "training/train_tfnet_gpu.py"
        "training/configs/gpu_config.json"
        "training/tfnet_model.py"
        "training/config_manager.py"
        "training/data_processor.py"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            error "Missing required file: $file"
            return 1
        fi
    done
    
    success "Project structure OK"
    return 0
}

# Function to create necessary directories
create_directories() {
    log "Creating GPU training directories..."
    
    directories=(
        "training/checkpoints_gpu"
        "training/logs_gpu"
        "training/output_gpu"
        "training/graphs"
    )
    
    for dir in "${directories[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            success "Created directory: $dir"
        fi
    done
    
    success "All directories ready"
    return 0
}

# Function to set environment variables for optimal GPU performance
set_gpu_env() {
    log "Setting GPU environment variables..."
    
    # Use first GPU
    export CUDA_VISIBLE_DEVICES=0
    
    # Clear any distributed training settings
    export MINDSPORE_HCCL_CONF_FILE=""
    
    # Set memory growth to avoid OOM
    export CUDA_MEMORY_FRACTION=0.8
    
    # Enable GPU memory optimization
    export MINDSPORE_GPU_MEMORY_OPTIMIZE=1
    
    success "GPU environment variables set"
}

# Function to run the training
run_training() {
    local config_file=${1:-"configs/gpu_config.json"}
    local dry_run=${2:-false}
    
    log "Starting GPU training..."
    log "Config file: $config_file"
    
    # Change to training directory
    cd training
    
    # Prepare command
    cmd="python start_training_gpu.py --config $config_file"
    
    if [[ "$dry_run" == "true" ]]; then
        cmd="$cmd --dry-run"
    fi
    
    log "Executing: $cmd"
    log "Working directory: $(pwd)"
    
    # Run the command
    if eval "$cmd"; then
        success "Training completed successfully!"
        return 0
    else
        error "Training failed!"
        return 1
    fi
}

# Function to display help
show_help() {
    cat << EOF
GPU-Optimized TFNet Training Launcher

Usage: $0 [OPTIONS]

OPTIONS:
    -c, --config FILE     Use specific config file (default: configs/gpu_config.json)
    -n, --dry-run        Show what would be executed without running
    -f, --force          Skip environment checks and force execution
    -s, --skip-checks    Skip all environment checks
    -h, --help           Show this help message

EXAMPLES:
    $0                           # Run with default config
    $0 -c custom_config.json     # Run with custom config
    $0 --dry-run                 # Test without actual execution
    $0 --force                   # Force run ignoring check failures

ENVIRONMENT:
    Requires conda environment 'mindspore-gpu' to be activated:
    conda activate mindspore-gpu

EOF
}

# Main function
main() {
    local config_file="configs/gpu_config.json"
    local dry_run=false
    local force=false
    local skip_checks=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -c|--config)
                config_file="$2"
                shift 2
                ;;
            -n|--dry-run)
                dry_run=true
                shift
                ;;
            -f|--force)
                force=true
                shift
                ;;
            -s|--skip-checks)
                skip_checks=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    log "GPU-Optimized TFNet Training Launcher"
    log "======================================"
    
    # Run environment checks unless skipped
    if [[ "$skip_checks" != "true" ]]; then
        log "Running environment checks..."
        
        check_failed=false
        
        if ! check_conda_env; then
            check_failed=true
        fi
        
        if ! check_gpu; then
            check_failed=true
        fi
        
        if ! check_mindspore; then
            check_failed=true
        fi
        
        if ! check_disk_space; then
            check_failed=true
        fi
        
        if ! check_project_structure; then
            check_failed=true
        fi
        
        if ! create_directories; then
            check_failed=true
        fi
        
        if [[ "$check_failed" == "true" && "$force" != "true" ]]; then
            error "Environment checks failed!"
            error "Please fix the issues above or use --force to proceed anyway."
            exit 1
        fi
        
        if [[ "$check_failed" == "true" && "$force" == "true" ]]; then
            warning "Proceeding despite failed checks (--force enabled)"
        fi
        
        success "Environment checks completed"
    else
        log "Environment checks skipped"
    fi
    
    # Set GPU environment
    set_gpu_env
    
    # Run training
    if run_training "$config_file" "$dry_run"; then
        success "Training launcher completed successfully!"
        exit 0
    else
        error "Training launcher failed!"
        exit 1
    fi
}

# Execute main function with all arguments
main "$@"
