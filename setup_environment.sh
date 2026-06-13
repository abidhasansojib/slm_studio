#!/bin/bash

# =================================================================
#  Mobile SLM Studio: Termux Environment Setup Script
# =================================================================

set -e  # Exit on error

echo "--------------------------------------------------------"
echo "  🚀 Initializing Mobile SLM Studio Environment Setup"
echo "--------------------------------------------------------"

# 1. Update and Upgrade System Packages
echo "[1/5] Updating Termux system packages..."
pkg update -y && pkg upgrade -y

# 2. Install Python and Essential Build Tools
echo "[2/5] Installing Python and compilation toolchain..."
pkg install python clang make ndk-sysroot build-essential libffi openssl -y

# 3. Upgrade Pip
echo "[3/5] Upgrading Python package manager (pip)..."
python3 -m pip install --upgrade pip

# 4. Install PyTorch and Flask
echo "[4/4] Installing core dependencies (PyTorch, Flask)..."
echo "NOTE: PyTorch installation in Termux may take several minutes."
pip install torch flask numpy

echo "--------------------------------------------------------"
echo "  ✅ Environment Setup Complete!"
echo "--------------------------------------------------------"
echo "To start the Web Dashboard, run:"
echo "  python server.py"
echo ""
echo "To start CLI training, run:"
echo "  python model.py"
echo "--------------------------------------------------------"
