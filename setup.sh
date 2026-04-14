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

is_torch_compatible_python() {
    "$1" -c "import sys; raise SystemExit(0 if ((3, 9) <= sys.version_info[:2] <= (3, 13)) else 1)" >/dev/null 2>&1
}

get_python_version() {
    "$1" -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}')"
}

create_venv() {
    local candidate venv_python selected_ver

    # On Windows, prefer specific Python versions via the Python launcher.
    if command -v py >/dev/null 2>&1; then
        # Prefer 3.12 first (validated with current SAM3 + CUDA setup).
        for selected_ver in 3.12 3.13 3.11 3.10 3.9; do
            echo "  Trying virtualenv with: py -$selected_ver"
            rm -rf .venv
            if py -"$selected_ver" -m venv .venv; then
                if [ -f .venv/bin/python ]; then
                    venv_python=".venv/bin/python"
                elif [ -f .venv/Scripts/python.exe ]; then
                    venv_python=".venv/Scripts/python.exe"
                else
                    continue
                fi

                if is_torch_compatible_python "$venv_python"; then
                    PYTHON_CMD="py -$selected_ver"
                    return 0
                fi
            fi
        done
    fi

    for candidate in python3 python python.exe; do
        if command -v "$candidate" >/dev/null 2>&1; then
            echo "  Trying virtualenv with: $candidate"
            rm -rf .venv
            if "$candidate" -m venv .venv; then
                if [ -f .venv/bin/python ]; then
                    venv_python=".venv/bin/python"
                elif [ -f .venv/Scripts/python.exe ]; then
                    venv_python=".venv/Scripts/python.exe"
                else
                    continue
                fi

                if is_torch_compatible_python "$venv_python"; then
                    PYTHON_CMD="$candidate"
                    return 0
                fi

                echo "    Skipping incompatible Python ($(get_python_version "$venv_python")); PyTorch wheels are unavailable for this version."
            fi
        fi
    done
    return 1
}

echo ""
echo "[1/5] Creating Python virtual environment..."
if ! create_venv; then
    echo "ERROR: Could not create a PyTorch-compatible virtual environment."
    echo "Install Python 3.13/3.12/3.11, then rerun setup.sh."
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

if ! is_torch_compatible_python "$VENV_PYTHON"; then
    echo "ERROR: Selected Python version ($(get_python_version "$VENV_PYTHON")) is not supported by PyTorch wheels."
    echo "Install Python 3.13/3.12/3.11 and rerun setup.sh."
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
elif [ "${FORCE_CUDA:-0}" = "1" ]; then
    echo "  FORCE_CUDA=1 - installing stable PyTorch with CUDA 12.4"
    "$VENV_PYTHON" -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
elif "$VENV_PYTHON" -c "import subprocess; raise SystemExit(0 if subprocess.run(['nvidia-smi'], capture_output=True).returncode == 0 else 1)" 2>/dev/null; then
    echo "  GPU detected - installing PyTorch nightly with CUDA 13.0"
    echo "  (recommended for newest RTX 50-series cards)"
    "$VENV_PYTHON" -m pip install --pre --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130
else
    echo "  No GPU detected - installing CPU-only PyTorch"
    "$VENV_PYTHON" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

echo ""
echo "[4/5] Installing transformers from source (main branch)..."
"$VENV_PYTHON" -m pip install git+https://github.com/huggingface/transformers.git

echo ""
echo "[5/5] Installing remaining dependencies..."
"$VENV_PYTHON" -m pip install -r requirements.txt

echo ""
echo "============================================"
echo " Setup complete!"
echo ""
echo " To activate the environment:"
echo "   Linux/macOS: source .venv/bin/activate"
echo "   Windows Bash: source .venv/Scripts/activate"
echo "   Windows PowerShell: .\\.venv\\Scripts\\Activate.ps1"
echo ""
echo " Next: run  python login.py  to authenticate"
echo "       with your HuggingFace token."
echo "============================================"
