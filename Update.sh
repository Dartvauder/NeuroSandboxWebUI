#!/bin/bash

git pull

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

source "$CURRENT_DIR/venv/bin/activate"

echo "Updating dependencies..."
python3 -m pip install --upgrade pip
pip install wheel setuptools
pip install --no-deps -r "$CURRENT_DIR/RequirementsFiles/requirements.txt"
pip install --no-deps -r "$CURRENT_DIR/RequirementsFiles/requirements-cuda.txt"
pip install --no-deps -r "$CURRENT_DIR/RequirementsFiles/requirements-llama-cpp.txt"
pip install git+https://github.com/tencent-ailab/IP-Adapter.git
pip install git+https://github.com/vork/PyNanoInstantMeshes.git
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/tatsy/torchmcubes.git
clear

echo "Post-installing patches..."
python3 "$CURRENT_DIR%RequirementsFiles\post_install.py"
clear

echo "Application has been updated successfully. Run start.sh"

deactivate

read -p "Press enter to continue"
