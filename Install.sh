#!/bin/bash

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

echo "Creating virtual environment..."
python3 -m venv "$CURRENT_DIR/venv"
source "$CURRENT_DIR/venv/bin/activate"
clear

echo "Upgrading pip, setuptools and wheel..."
python3 -m pip install --upgrade pip==24.0
pip install wheel setuptools
sleep 3
clear

echo "Installing dependencies..."
mkdir -p "$CURRENT_DIR/logs"
ERROR_LOG="$CURRENT_DIR/logs/installation_errors.log"
touch "$ERROR_LOG"

pip install --no-deps -r "$CURRENT_DIR/RequirementsFiles/requirements.txt" 2>> "$ERROR_LOG"
pip install --no-deps -r "$CURRENT_DIR/RequirementsFiles/requirements-cuda.txt" 2>> "$ERROR_LOG"
pip install --no-deps -r "$CURRENT_DIR/RequirementsFiles/requirements-llama-cpp.txt" 2>> "$ERROR_LOG"
pip install git+https://github.com/tencent-ailab/IP-Adapter.git 2>> "$ERROR_LOG"
pip install git+https://github.com/vork/PyNanoInstantMeshes.git 2>> "$ERROR_LOG"
pip install git+https://github.com/openai/CLIP.git 2>> "$ERROR_LOG"
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