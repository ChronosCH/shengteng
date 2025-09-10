#!/bin/bash

# Quick Start Script for TFNet Training (Linux)
# Simplified version for experienced users

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo "============================================================"
echo "TFNet Quick Training Launcher (Linux)"
echo "============================================================"

# Check if we're in the right directory
if [[ ! -d "training" ]]; then
    echo -e "${RED}Error: Run this script from project root directory${NC}"
    exit 1
fi

# Activate conda environment
echo "Activating conda environment..."
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || \
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || {
    echo -e "${RED}Error: Could not find conda initialization script${NC}"
    echo "Please run: source /path/to/conda/etc/profile.d/conda.sh"
    exit 1
}

conda activate shengteng || {
    echo -e "${RED}Error: Failed to activate 'shengteng' environment${NC}"
    echo "Please create environment: conda create -n shengteng python=3.8"
    exit 1
}

echo -e "${GREEN}Environment activated successfully${NC}"

# Default action
ACTION="${1:-train}"

case "$ACTION" in
    "train")
        echo "Starting training..."
        python training/train_tfnet.py --config training/configs/tfnet_config.json
        ;;
    "eval")
        echo "Starting evaluation..."
        python training/evaluator.py --config training/configs/tfnet_config.json
        ;;
    "test")
        echo "Running basic test..."
        python training/test_basic.py
        ;;
    *)
        echo "Usage: $0 [train|eval|test]"
        echo "  train - Start training (default)"
        echo "  eval  - Run evaluation"
        echo "  test  - Run basic test"
        exit 1
        ;;
esac

echo -e "${GREEN}Operation completed!${NC}"
