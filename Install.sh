#!/bin/bash

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

echo "Creating virtual environment..."
python3 -m venv "$CURRENT_DIR/venv"
source "$CURRENT_DIR/venv/bin/activate"
clear

echo "Upgrading pip, setuptools and whell..."
python3 -m pip install --upgrade pip
pip install wheel setuptools
clear

echo "Installing dependencies..."
pip install --no-deps -r "$CURRENT_DIR/RequirementsFiles/requirements.txt"
pip install --no-deps -r "$CURRENT_DIR/RequirementsFiles/requirements-cuda.txt"
pip install --no-deps -r "$CURRENT_DIR/RequirementsFiles/requirements-llama-cpp.txt"
pip install git+https://github.com/vork/PyNanoInstantMeshes.git
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/tatsy/torchmcubes.git
clear

echo "Application has been installed successfully. Run start.sh"

deactivate

read -p "Press enter to continue"
