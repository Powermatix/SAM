#!/usr/bin/env bash
set -e

echo "============================================"
echo " SAM3 Environment Setup"
echo "============================================"

# ── 1. Create virtual environment ────────────────────────────────────────────
echo ""
echo "[1/5] Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# ── 2. Upgrade pip ────────────────────────────────────────────────────────────
echo ""
echo "[2/5] Upgrading pip..."
pip install --upgrade pip setuptools wheel

# ── 3. Install PyTorch ────────────────────────────────────────────────────────
echo ""
echo "[3/5] Installing PyTorch..."
# Auto-detect CUDA
if python3 -c "import subprocess; r = subprocess.run(['nvidia-smi'], capture_output=True); exit(0 if r.returncode == 0 else 1)" 2>/dev/null; then
    echo "  GPU detected — installing PyTorch with CUDA 12.4"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
else
    echo "  No GPU detected — installing CPU-only PyTorch"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# ── 4. Install transformers from source (required for SAM3) ──────────────────
echo ""
echo "[4/5] Installing transformers from source (main branch)..."
pip install git+https://github.com/huggingface/transformers.git

# ── 5. Install remaining dependencies ────────────────────────────────────────
echo ""
echo "[5/5] Installing remaining dependencies..."
pip install Pillow requests matplotlib numpy huggingface_hub

echo ""
echo "============================================"
echo " Setup complete!"
echo ""
echo " To activate the environment:"
echo "   source .venv/bin/activate"
echo ""
echo " Next: run  python login.py  to authenticate"
echo "       with your HuggingFace token."
echo "============================================"
