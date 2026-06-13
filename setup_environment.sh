#!/bin/bash

# =================================================================
#  Mobile SLM Studio: Termux Environment Setup Script (ARM64)
# =================================================================

set -e  # Exit on error

echo "--------------------------------------------------------"
echo "  🚀 Initializing Mobile SLM Studio Environment Setup"
echo "  Target: Android Termux (ARM64)"
echo "--------------------------------------------------------"

# 1. Update and Upgrade System Packages
echo "[1/4] Updating Termux system packages..."
pkg update -y && pkg upgrade -y

# 2. Install Python, C++ Compiler, and Llama.cpp Engine
echo "[2/4] Installing Python, compiler toolchain, and llama-cpp engine..."
pkg install python clang make ndk-sysroot build-essential libffi openssl llama-cpp -y

# 3. Upgrade Pip
# 4. Install PyTorch, Flask, and Numpy
echo "[4/4] Installing core dependencies (PyTorch, Flask, Numpy)..."
echo "NOTE: PyTorch installation in Termux may take several minutes to build/install."
pip install numpy flask torch

echo "--------------------------------------------------------"
echo "  ✅ Environment Setup Complete!"
echo "--------------------------------------------------------"
echo "  💡 NOTE: We have integrated a pure Python implementation of"
echo "     Safetensors (loading & saving) so you DO NOT need to"
echo "     install the 'safetensors' pip module (which requires Rust)."
echo "--------------------------------------------------------"
echo "To start the Web Dashboard, run:"
echo "  python server.py"
echo ""
echo "To start CLI training, run:"
echo "  python model.py"
echo "--------------------------------------------------------"
