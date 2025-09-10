#!/bin/bash

# TFNet Environment Setup Script for Linux
# One-click installation and configuration

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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
    echo "TFNet Environment Setup Script"
    echo "Continuous Sign Language Recognition System"
    echo "============================================================"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install conda if not present
install_conda() {
    print_info "Installing Miniconda..."
    
    # Detect architecture
    ARCH=$(uname -m)
    if [[ "$ARCH" == "x86_64" ]]; then
        CONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    elif [[ "$ARCH" == "aarch64" ]]; then
        CONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"
    else
        print_error "Unsupported architecture: $ARCH"
        return 1
    fi
    
    # Download and install
    CONDA_INSTALLER="/tmp/miniconda_installer.sh"
    print_info "Downloading Miniconda from $CONDA_URL"
    wget -O "$CONDA_INSTALLER" "$CONDA_URL" || {
        print_error "Failed to download Miniconda"
        return 1
    }
    
    chmod +x "$CONDA_INSTALLER"
    print_info "Installing Miniconda to $HOME/miniconda3"
    bash "$CONDA_INSTALLER" -b -p "$HOME/miniconda3" || {
        print_error "Failed to install Miniconda"
        return 1
    }
    
    # Initialize conda
    print_info "Initializing conda..."
    "$HOME/miniconda3/bin/conda" init bash
    source "$HOME/.bashrc" 2>/dev/null || true
    
    # Clean up
    rm -f "$CONDA_INSTALLER"
    
    print_success "Miniconda installed successfully"
    return 0
}

# Setup conda environment
setup_conda_env() {
    print_info "Setting up conda environment 'shengteng'..."
    
    # Source conda
    if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    else
        print_error "Could not find conda initialization script"
        return 1
    fi
    
    # Remove existing environment if it exists
    if conda env list | grep -q "shengteng"; then
        print_warning "Environment 'shengteng' already exists, removing..."
        conda env remove -n shengteng -y
    fi
    
    # Create new environment
    print_info "Creating conda environment with Python 3.8..."
    conda create -n shengteng python=3.8 -y
    
    # Activate environment
    conda activate shengteng
    
    print_success "Conda environment created and activated"
    return 0
}

# Install Python dependencies
install_dependencies() {
    print_info "Installing Python dependencies..."
    
    # Core dependencies
    print_info "Installing core packages..."
    pip install mindspore>=1.8.0 || {
        print_warning "Failed to install mindspore, trying alternative..."
        pip install mindspore-cpu>=1.8.0
    }
    
    pip install opencv-python>=4.5.0
    pip install numpy>=1.19.0
    pip install pandas>=1.2.0
    pip install matplotlib>=3.3.0
    pip install tqdm>=4.60.0
    pip install scikit-learn>=0.24.0
    
    # Optional dependencies for enhanced functionality
    print_info "Installing optional packages..."
    pip install seaborn tensorboard || print_warning "Some optional packages failed to install"
    
    print_success "Dependencies installed successfully"
    return 0
}

# Set up project permissions
setup_permissions() {
    print_info "Setting up script permissions..."
    
    # Make scripts executable
    chmod +x training/start_training.sh 2>/dev/null || true
    chmod +x training/quick_start.sh 2>/dev/null || true
    chmod +x setup_environment.sh 2>/dev/null || true
    
    # Create necessary directories
    mkdir -p training/logs
    mkdir -p training/checkpoints
    mkdir -p training/output
    mkdir -p data/CE-CSL/video/train
    mkdir -p data/CE-CSL/video/dev
    mkdir -p data/CE-CSL/label
    
    print_success "Permissions and directories set up"
    return 0
}

# Verify installation
verify_installation() {
    print_info "Verifying installation..."
    
    # Check Python packages
    python -c "import mindspore; print(f'MindSpore version: {mindspore.__version__}')" || {
        print_error "MindSpore verification failed"
        return 1
    }
    
    python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')" || {
        print_error "OpenCV verification failed"
        return 1
    }
    
    python -c "import numpy; print(f'NumPy version: {numpy.__version__}')" || {
        print_error "NumPy verification failed"
        return 1
    }
    
    print_success "All packages verified successfully"
    return 0
}

# Create activation script
create_activation_script() {
    print_info "Creating environment activation script..."
    
    cat > activate_training_env.sh << 'EOF'
#!/bin/bash
# Quick environment activation script

# Source conda
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "Error: Could not find conda initialization script"
    exit 1
fi

# Activate environment
conda activate shengteng

echo "TFNet training environment activated!"
echo "You can now run: ./training/start_training.sh"
EOF
    
    chmod +x activate_training_env.sh
    
    print_success "Activation script created: activate_training_env.sh"
}

# Display usage instructions
show_usage_instructions() {
    print_success "Setup completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Activate the environment:"
    echo "   source activate_training_env.sh"
    echo ""
    echo "2. Start training:"
    echo "   ./training/start_training.sh train"
    echo ""
    echo "3. Or use quick start:"
    echo "   ./training/quick_start.sh"
    echo ""
    echo "For detailed instructions, see:"
    echo "   training/README_LINUX_SCRIPTS.md"
    echo ""
    print_info "Configuration files:"
    echo "   - Training config: training/configs/tfnet_config.json"
    echo "   - Logs directory: training/logs/"
    echo "   - Checkpoints: training/checkpoints/"
}

# Main execution
main() {
    print_header
    
    # Check if we're in the right directory
    if [[ ! -d "training" ]]; then
        print_error "Please run this script from the project root directory"
        print_error "Current directory: $(pwd)"
        exit 1
    fi
    
    # Check for conda
    if ! command_exists conda; then
        print_warning "Conda not found, installing Miniconda..."
        if ! command_exists wget; then
            print_error "wget is required for installation. Please install wget first:"
            print_error "  sudo apt-get install wget  # On Ubuntu/Debian"
            print_error "  sudo yum install wget      # On CentOS/RHEL"
            exit 1
        fi
        install_conda
        print_info "Please restart your terminal or run: source ~/.bashrc"
        print_info "Then run this script again to continue setup."
        exit 0
    fi
    
    # Setup environment
    if setup_conda_env; then
        print_success "Conda environment setup completed"
    else
        print_error "Failed to setup conda environment"
        exit 1
    fi
    
    # Install dependencies
    if install_dependencies; then
        print_success "Dependencies installation completed"
    else
        print_error "Failed to install dependencies"
        exit 1
    fi
    
    # Setup permissions and directories
    setup_permissions
    
    # Verify installation
    if verify_installation; then
        print_success "Installation verification completed"
    else
        print_error "Installation verification failed"
        exit 1
    fi
    
    # Create activation script
    create_activation_script
    
    # Show usage instructions
    show_usage_instructions
}

# Run main function
main "$@"
