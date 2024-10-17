#!/bin/bash

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

echo "Creating virtual environment..."
python3 -m venv "$CURRENT_DIR/venv"
source "$CURRENT_DIR/venv/bin/activate"
clear

echo "Setting up local pip cache..."
mkdir -p "$CURRENT_DIR/pip_cache"
export PIP_CACHE_DIR="$CURRENT_DIR/pip_cache"

echo "Upgrading pip, setuptools and wheel..."
python3 -m pip install --upgrade pip
pip install wheel setuptools
sleep 3
clear

echo "Installing dependencies..."
mkdir -p "$CURRENT_DIR/logs"
ERROR_LOG="$CURRENT_DIR/logs/installation_errors.log"
touch "$ERROR_LOG"

export BUILD_CUDA_EXT=1
export INSTALL_KERNELS=1

pip install --no-deps -r "$CURRENT_DIR/RequirementsFiles/requirements.txt" 2>> "$ERROR_LOG"
pip install --no-deps -r "$CURRENT_DIR/RequirementsFiles/requirements-cuda.txt" 2>> "$ERROR_LOG"
pip install --no-deps -r "$CURRENT_DIR/RequirementsFiles/requirements-llama-cpp.txt" 2>> "$ERROR_LOG"
pip install --no-deps -r "$CURRENT_DIR/RequirementsFiles/requirements-stable-diffusion-cpp.txt" 2>> "$ERROR_LOG"
pip install --no-build-isolation -e git+https://github.com/PanQiWei/AutoGPTQ.git#egg=auto_gptq 2>> "$ERROR_LOG"
pip install triton==2.1.0 2>> "$ERROR_LOG"
pip install --no-build-isolation -e git+https://github.com/casper-hansen/AutoAWQ.git#egg=autoawq 2>> "$ERROR_LOG"
pip install --no-build-isolation -e git+https://github.com/turboderp/exllamav2.git#egg=exllamav2 2>> "$ERROR_LOG"
pip install git+https://github.com/tencent-ailab/IP-Adapter.git 2>> "$ERROR_LOG"
pip install git+https://github.com/vork/PyNanoInstantMeshes.git 2>> "$ERROR_LOG"
pip install git+https://github.com/openai/CLIP.git 2>> "$ERROR_LOG"
pip install git+https://github.com/xhinker/sd_embed.git@main 2>> "$ERROR_LOG"
sleep 3
clear

echo "Post-installing patches..."
python3 "$CURRENT_DIR/RequirementsFiles/post_install.py"
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