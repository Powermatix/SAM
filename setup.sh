#!/usr/bin/env bash
set -euo pipefail

echo "============================================"
echo " SAM3 Environment Setup"
echo "============================================"

detect_python() {
    local candidate
    for candidate in python3 python python.exe; do
        if command -v "$candidate" >/dev/null 2>&1; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

create_venv() {
    local candidate
    for candidate in python3 python python.exe; do
        if command -v "$candidate" >/dev/null 2>&1; then
            echo "  Trying virtualenv with: $candidate"
            rm -rf .venv
            if "$candidate" -m venv .venv; then
                PYTHON_CMD="$candidate"
                return 0
            fi
        fi
    done
    return 1
}

echo ""
echo "[1/5] Creating Python virtual environment..."
if ! create_venv; then
    echo "ERROR: Could not create virtual environment with python3/python/python.exe"
    exit 1
fi

if [ -f .venv/bin/python ]; then
    VENV_PYTHON=".venv/bin/python"
elif [ -f .venv/Scripts/python.exe ]; then
    VENV_PYTHON=".venv/Scripts/python.exe"
else
    echo "ERROR: Virtualenv Python executable not found."
    exit 1
fi

echo ""
echo "[2/5] Upgrading pip..."
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel

echo ""
echo "[3/5] Installing PyTorch..."
if [ "${FORCE_CPU:-0}" = "1" ]; then
    echo "  FORCE_CPU=1 - installing CPU-only PyTorch"
    "$VENV_PYTHON" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
elif "$VENV_PYTHON" -c "import subprocess; raise SystemExit(0 if subprocess.run(['nvidia-smi'], capture_output=True).returncode == 0 else 1)" 2>/dev/null; then
    echo "  GPU detected - installing PyTorch with CUDA 12.4"
    "$VENV_PYTHON" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
else
    echo "  No GPU detected - installing CPU-only PyTorch"
    "$VENV_PYTHON" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

echo ""
echo "[4/5] Installing transformers from source (main branch)..."
"$VENV_PYTHON" -m pip install git+https://github.com/huggingface/transformers.git

echo ""
echo "[5/5] Installing remaining dependencies..."
"$VENV_PYTHON" -m pip install Pillow requests matplotlib numpy huggingface_hub

echo ""
echo "============================================"
echo " Setup complete!"
echo ""
echo " To activate the environment:"
echo "   Linux/macOS: source .venv/bin/activate"
echo "   Windows Bash: source .venv/Scripts/activate"
echo ""
echo " Next: run  python login.py  to authenticate"
echo "       with your HuggingFace token."
echo "============================================"
