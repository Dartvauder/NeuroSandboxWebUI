#!/bin/bash

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

echo "Select installation type:"
echo "1. GPU"
echo "2. CPU OR MPS"
read -n 1 -p "Enter number (1 or 2): " choice
echo ""

if [ "$choice" = "2" ]; then
    INSTALL_TYPE="CPU"
    export BUILD_CUDA_EXT=0
    export INSTALL_KERNELS=0

    if system_profiler SPDisplaysDataType | grep -q "Metal"; then
        echo "MPS is detected. Installing MPS-specific requirements."
        MPS_MODE=true
    else
        MPS_MODE=false
    fi
else
    INSTALL_TYPE="GPU"
    export BUILD_CUDA_EXT=1
    export INSTALL_KERNELS=1
    MPS_MODE=false
fi

clear
echo "Selected version: $INSTALL_TYPE"
sleep 2
clear

echo "Creating virtual environment..."
python -m venv "$CURRENT_DIR/venv"
source "$CURRENT_DIR/venv/bin/activate"
clear

echo "Setting up local pip cache..."
mkdir -p "$CURRENT_DIR/TechnicalFiles/pip_cache"
export PIP_CACHE_DIR="$CURRENT_DIR/TechnicalFiles/pip_cache"

echo "Upgrading pip, setuptools and wheel..."
python -m pip install --upgrade pip setuptools
pip install wheel
sleep 3
clear

echo "Installing dependencies..."
mkdir -p "$CURRENT_DIR/TechnicalFiles/logs"
ERROR_LOG="$CURRENT_DIR/TechnicalFiles/logs/installation_errors.log"
touch "$ERROR_LOG"

if [ "$INSTALL_TYPE" = "CPU" ]; then
    if [ "$MPS_MODE" = true ]; then
        pip install --no-deps -r "$CURRENT_DIR/RequirementsFiles/requirements-СPU.txt" 2>> "$ERROR_LOG"
        pip install --no-deps -r "$CURRENT_DIR/RequirementsFiles/requirements-cuda-CPU.txt" 2>> "$ERROR_LOG"
        pip install --no-deps -r "$CURRENT_DIR/RequirementsFiles/requirements-llama-cpp-MPS.txt" 2>> "$ERROR_LOG"
        pip install --no-deps -r "$CURRENT_DIR/RequirementsFiles/requirements-stable-diffusion-cpp-MPS.txt" 2>> "$ERROR_LOG"
    else
        pip install --no-deps -r "$CURRENT_DIR/RequirementsFiles/requirements-СPU.txt" 2>> "$ERROR_LOG"
        pip install --no-deps -r "$CURRENT_DIR/RequirementsFiles/requirements-cuda-CPU.txt" 2>> "$ERROR_LOG"
        pip install --no-deps -r "$CURRENT_DIR/RequirementsFiles/requirements-llama-cpp-CPU.txt" 2>> "$ERROR_LOG"
        pip install --no-deps -r "$CURRENT_DIR/RequirementsFiles/requirements-stable-diffusion-cpp-CPU.txt" 2>> "$ERROR_LOG"
    fi
else
    pip install --no-deps -r "$CURRENT_DIR/RequirementsFiles/requirements.txt" 2>> "$ERROR_LOG"
    pip install --no-deps -r "$CURRENT_DIR/RequirementsFiles/requirements-cuda.txt" 2>> "$ERROR_LOG"
    pip install --no-deps -r "$CURRENT_DIR/RequirementsFiles/requirements-llama-cpp.txt" 2>> "$ERROR_LOG"
    pip install --no-deps -r "$CURRENT_DIR/RequirementsFiles/requirements-stable-diffusion-cpp.txt" 2>> "$ERROR_LOG"
fi
echo "INSTALL_TYPE=$INSTALL_TYPE" > "$CURRENT_DIR/TechnicalFiles/install_config.txt"

pip install triton==3.0.0 2>> "$ERROR_LOG"
pip install --no-build-isolation -e git+https://github.com/PanQiWei/AutoGPTQ.git#egg=auto_gptq@v0.7.1 2>> "$ERROR_LOG"
pip install --no-build-isolation -e git+https://github.com/casper-hansen/AutoAWQ.git#egg=autoawq@v0.2.6 2>> "$ERROR_LOG"
pip install --no-build-isolation -e git+https://github.com/turboderp/exllamav2.git#egg=exllamav2@v0.2.3 2>> "$ERROR_LOG"
pip install git+https://github.com/tencent-ailab/IP-Adapter.git 2>> "$ERROR_LOG"
pip install git+https://github.com/vork/PyNanoInstantMeshes.git 2>> "$ERROR_LOG"
pip install git+https://github.com/openai/CLIP.git 2>> "$ERROR_LOG"
pip install git+https://github.com/xhinker/sd_embed.git@main 2>> "$ERROR_LOG"
sleep 3
clear

echo "Post-installing patches..."
python "$CURRENT_DIR/TechnicalFiles/post_install.py"
sleep 3
clear

echo "Checking for installation errors..."
if grep -iq "error" "$ERROR_LOG"; then
    echo "Some packages failed to install. Please check $ERROR_LOG for details."
else
    echo "All packages installed successfully."
fi
sleep 5
clear

echo "Application installation process completed. Run start.sh to launch the application."

deactivate

read -p "Press enter to continue"