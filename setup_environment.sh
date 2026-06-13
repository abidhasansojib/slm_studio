#!/bin/bash

# =================================================================
#  SLM Studio: Termux Environment Setup Script (ARM64)
# =================================================================

set -e  # Exit on error

echo "--------------------------------------------------------"
echo "  🚀 Initializing SLM Studio Environment Setup"
echo "  Target: Android Termux (ARM64)"
echo "--------------------------------------------------------"

# 1. Update and Upgrade System Packages
echo "[1/4] Updating Termux system packages..."
pkg update -y && pkg upgrade -y

# 2. Install Python, C++ Compiler, and Llama.cpp Engine
echo "[2/4] Installing Python, compiler toolchain, and essential libraries..."
pkg install python clang make ndk-sysroot build-essential libffi openssl -y

echo "[*] Adding Termux User Repository (TUR) for extra packages..."
pkg install tur-repo -y || true
pkg update -y

echo "[*] Installing pre-compiled math libraries (Numpy/Torch)..."
pkg install python-numpy python-torch -y

echo "[*] Attempting to install llama-cpp package..."
pkg install llama-cpp -y || echo "[-] Warning: llama-cpp package not found in repositories. GGUF inference will require manual compilation of llama.cpp."

# 3. Install Flask and GGUF tools
echo "[3/4] Installing Flask and GGUF support via Pip..."
pip install flask gguf

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
