#!/bin/bash

# TFNet Training Startup Script for Linux
# Optimized for CPU execution with MindSpore
# Continuous Sign Language Recognition System

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
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
    echo "TFNet Training System"
    echo "Continuous Sign Language Recognition"
    echo "Platform: Linux $(uname -r)"
    echo "============================================================"
}

# Parse command line arguments
ACTION="${1:-train}"
CONFIG_PATH="${2:-training/configs/tfnet_config.json}"
RESUME_PATH="$3"
MODEL_PATH="$4"
SKIP_CHECKS="false"

# Parse additional options
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-checks)
            SKIP_CHECKS="true"
            shift
            ;;
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --resume)
            RESUME_PATH="$2"
            shift 2
            ;;
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [ACTION] [OPTIONS]"
            echo ""
            echo "Actions:"
            echo "  train    Start training (default)"
            echo "  eval     Run evaluation"
            echo "  check    Check environment only"
            echo ""
            echo "Options:"
            echo "  --config PATH     Configuration file path (default: training/configs/tfnet_config.json)"
            echo "  --resume PATH     Resume training from checkpoint"
            echo "  --model PATH      Model path for evaluation"
            echo "  --skip-checks     Skip environment checks"
            echo "  --help, -h        Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 train                          # Start training with default config"
            echo "  $0 train --config custom.json    # Start training with custom config"
            echo "  $0 eval --model best_model.ckpt   # Run evaluation"
            echo "  $0 check                          # Check environment only"
            exit 0
            ;;
        *)
            if [[ -z "$ACTION" ]]; then
                ACTION="$1"
            fi
            shift
            ;;
    esac
done

print_header
print_info "Action: $ACTION"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check conda environment
check_conda_env() {
    print_info "Checking conda environment..."
    
    if ! command_exists conda; then
        print_error "Conda not found. Please install conda and add it to PATH."
        return 1
    fi
    
    # Check if shengteng environment exists
    if ! conda env list | grep -q "shengteng"; then
        print_error "Conda environment 'shengteng' not found."
        print_info "To create the environment:"
        print_info "  conda create -n shengteng python=3.8"
        print_info "  conda activate shengteng"
        print_info "  pip install mindspore opencv-python numpy"
        return 1
    fi
    
    print_success "Conda environment check passed"
    return 0
}

# Function to activate conda environment
activate_conda_env() {
    print_info "Activating conda environment 'shengteng'..."
    
    # Source conda initialization
    if [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
        source "/opt/conda/etc/profile.d/conda.sh"
    else
        # Try to find conda.sh in common locations
        for conda_path in "/usr/local/anaconda3" "/usr/local/miniconda3" "/opt/anaconda3" "/opt/miniconda3"; do
            if [[ -f "$conda_path/etc/profile.d/conda.sh" ]]; then
                source "$conda_path/etc/profile.d/conda.sh"
                break
            fi
        done
    fi
    
    # Activate environment
    conda activate shengteng
    if [[ $? -ne 0 ]]; then
        print_error "Failed to activate conda environment 'shengteng'"
        return 1
    fi
    
    print_success "Conda environment activated"
    return 0
}

# Function to check project structure
check_project_structure() {
    print_info "Checking project structure..."
    
    # Check if we're in the correct directory
    if [[ ! -d "training" ]]; then
        print_error "Please run this script from the project root directory"
        print_error "Current directory: $(pwd)"
        print_error "Expected to find: training/ folder"
        return 1
    fi
    
    # Check essential training files
    local training_files=(
        "training/train_tfnet.py"
        "training/evaluator.py"
        "training/config_manager.py"
        "training/configs/tfnet_config.json"
    )
    
    local missing_files=()
    for file in "${training_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            missing_files+=("$file")
        fi
    done
    
    if [[ ${#missing_files[@]} -gt 0 ]]; then
        print_error "Missing essential training files:"
        for file in "${missing_files[@]}"; do
            print_error "  ✗ $file"
        done
        return 1
    fi
    
    print_success "Project structure OK"
    return 0
}

# Function to check dataset
check_dataset() {
    print_info "Checking dataset structure..."
    
    local data_paths=(
        "data/CE-CSL/video/train"
        "data/CE-CSL/video/dev"
        "data/CE-CSL/label/train.csv"
        "data/CE-CSL/label/dev.csv"
    )
    
    local missing_paths=()
    for path in "${data_paths[@]}"; do
        if [[ ! -e "$path" ]]; then
            missing_paths+=("$path")
        fi
    done
    
    if [[ ${#missing_paths[@]} -gt 0 ]]; then
        print_warning "Some dataset paths are missing:"
        for path in "${missing_paths[@]}"; do
            print_warning "  ✗ $path"
        done
        print_warning "Training may fail without proper dataset structure"
        return 1
    fi
    
    print_success "Dataset structure OK"
    return 0
}

# Function to check dependencies
check_dependencies() {
    print_info "Checking Python dependencies..."
    
    local required_packages=("mindspore" "cv2" "numpy")
    local missing_packages=()
    
    for package in "${required_packages[@]}"; do
        if ! python -c "import $package" 2>/dev/null; then
            missing_packages+=("$package")
        fi
    done
    
    if [[ ${#missing_packages[@]} -gt 0 ]]; then
        print_error "Missing required packages:"
        for package in "${missing_packages[@]}"; do
            print_error "  ✗ $package"
        done
        print_info "To install missing packages:"
        print_info "  conda activate shengteng"
        print_info "  pip install mindspore opencv-python numpy"
        return 1
    fi
    
    print_success "All dependencies available"
    return 0
}

# Function to run environment checks
run_environment_checks() {
    print_info "Running environment checks..."
    
    local checks_passed=true
    
    if ! check_project_structure; then
        checks_passed=false
    fi
    
    if ! check_conda_env; then
        checks_passed=false
    fi
    
    if ! activate_conda_env; then
        checks_passed=false
    fi
    
    if ! check_dependencies; then
        checks_passed=false
    fi
    
    check_dataset  # This is a warning-only check
    
    if [[ "$checks_passed" == "false" ]]; then
        print_error "Environment checks failed. Please fix the issues above."
        exit 1
    fi
    
    print_success "All environment checks passed!"
}

# Function to run training
run_training() {
    print_info "Starting TFNet training..."
    
    local cmd="python training/train_tfnet.py"
    
    if [[ -n "$CONFIG_PATH" ]]; then
        cmd="$cmd --config $CONFIG_PATH"
    fi
    
    if [[ -n "$RESUME_PATH" ]]; then
        cmd="$cmd --resume $RESUME_PATH"
    fi
    
    print_info "Running command: $cmd"
    
    # Create logs directory if it doesn't exist
    mkdir -p training/logs
    
    # Run training with output logging
    if eval "$cmd" 2>&1 | tee "training/logs/training_$(date +%Y%m%d_%H%M%S).log"; then
        print_success "Training completed successfully!"
        return 0
    else
        print_error "Training failed"
        return 1
    fi
}

# Function to run evaluation
run_evaluation() {
    print_info "Starting TFNet evaluation..."
    
    local cmd="python training/evaluator.py"
    
    if [[ -n "$CONFIG_PATH" ]]; then
        cmd="$cmd --config $CONFIG_PATH"
    fi
    
    if [[ -n "$MODEL_PATH" ]]; then
        cmd="$cmd --model $MODEL_PATH"
    fi
    
    print_info "Running command: $cmd"
    
    # Create logs directory if it doesn't exist
    mkdir -p training/logs
    
    # Run evaluation with output logging
    if eval "$cmd" 2>&1 | tee "training/logs/evaluation_$(date +%Y%m%d_%H%M%S).log"; then
        print_success "Evaluation completed successfully!"
        return 0
    else
        print_error "Evaluation failed"
        return 1
    fi
}

# Function to handle cleanup on exit
cleanup() {
    print_info "Cleaning up..."
    # Add any cleanup operations here if needed
}

# Set up signal handlers
trap cleanup EXIT INT TERM

# Main execution
main() {
    # Skip checks if requested
    if [[ "$SKIP_CHECKS" != "true" ]]; then
        run_environment_checks
    else
        print_warning "Skipping environment checks as requested"
        # Still need to activate conda environment
        if check_conda_env && activate_conda_env; then
            print_success "Conda environment activated"
        else
            print_error "Failed to activate conda environment"
            exit 1
        fi
    fi
    
    # Execute the requested action
    case "$ACTION" in
        "train")
            if run_training; then
                print_success "Training operation completed successfully!"
            else
                print_error "Training operation failed!"
                exit 1
            fi
            ;;
        "eval")
            if run_evaluation; then
                print_success "Evaluation operation completed successfully!"
            else
                print_error "Evaluation operation failed!"
                exit 1
            fi
            ;;
        "check")
            print_success "Environment check completed successfully!"
            ;;
        *)
            print_error "Unknown action: $ACTION"
            print_info "Valid actions: train, eval, check"
            exit 1
            ;;
    esac
    
    print_success "Operation completed!"
}

# Run main function
main "$@"
