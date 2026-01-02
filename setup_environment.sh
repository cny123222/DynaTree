#!/bin/bash
# Environment Setup Script for LLM Efficient Reasoning
# Usage: bash setup_environment.sh

set -e  # Exit on error

echo "=========================================="
echo "LLM Efficient Reasoning - Environment Setup"
echo "=========================================="

# Check Python version
echo ""
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Check if conda is available
if command -v conda &> /dev/null; then
    echo ""
    echo "Conda found. Using conda environment."
    
    # Check if environment exists
    if conda env list | grep -q "nlp"; then
        echo "Environment 'nlp' already exists."
        echo "Activating existing environment..."
        eval "$(conda shell.bash hook)"
        conda activate nlp
    else
        echo "Creating new conda environment 'nlp'..."
        conda create -n nlp python=3.11 -y
        eval "$(conda shell.bash hook)"
        conda activate nlp
    fi
else
    echo ""
    echo "Conda not found. Using system Python."
    echo "Consider using conda for better dependency management."
fi

# Install PyTorch (adjust for your CUDA version)
echo ""
echo "Installing PyTorch..."
echo "Detecting CUDA..."

if command -v nvidia-smi &> /dev/null; then
    cuda_version=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
    echo "CUDA $cuda_version detected"
    
    if [[ "$cuda_version" == "12."* ]]; then
        echo "Installing PyTorch for CUDA 12.x..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    elif [[ "$cuda_version" == "11."* ]]; then
        echo "Installing PyTorch for CUDA 11.x..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        echo "Installing PyTorch (default)..."
        pip install torch torchvision torchaudio
    fi
else
    echo "CUDA not detected. Installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other dependencies
echo ""
echo "Installing other dependencies..."
pip install -r requirements.txt

# Verify installation
echo ""
echo "=========================================="
echo "Verifying Installation"
echo "=========================================="

python3 << EOF
import sys
print(f"Python version: {sys.version}")

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"✗ PyTorch: {e}")

try:
    import transformers
    print(f"✓ Transformers {transformers.__version__}")
except ImportError as e:
    print(f"✗ Transformers: {e}")

try:
    import datasets
    print(f"✓ Datasets {datasets.__version__}")
except ImportError as e:
    print(f"✗ Datasets: {e}")

try:
    import matplotlib
    print(f"✓ Matplotlib {matplotlib.__version__}")
except ImportError as e:
    print(f"✗ Matplotlib: {e}")

print("")
print("All core dependencies installed successfully!")
EOF

# Create necessary directories
echo ""
echo "Creating necessary directories..."
mkdir -p results/final_experiments
mkdir -p papers/figures/final
mkdir -p models

echo ""
echo "=========================================="
echo "Environment Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Download models (if not already done):"
echo "   python download_pythia_2.8b.py"
echo "   python download_pythia_small.py"
echo ""
echo "2. Run experiments:"
echo "   bash run_experiments.sh"
echo ""

