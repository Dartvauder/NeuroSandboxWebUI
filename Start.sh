#!/bin/bash

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

source "$CURRENT_DIR/venv/bin/activate"

python "$CURRENT_DIR/TechnicalFiles/first_setup.py"

echo "Attempting to read Settings.json..."
if [ ! -f TechnicalFiles/Settings.json ]; then
    echo "Settings.json file not found!"
    exit 1
fi

echo "Contents of Settings.json:"
cat TechnicalFiles/Settings.json

HF_TOKEN=$(grep -oP '"hf_token"\s*:\s*"\K[^"]+' TechnicalFiles/Settings.json)

echo "HF_TOKEN value: '$HF_TOKEN'"

if [ -z "$HF_TOKEN" ]; then
    echo "HF token is empty or not found in Settings.json. Please add your Hugging Face token to this file."
    exit 1
fi

echo "Logging in to Hugging Face..."
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

echo "Launching NeuroSandboxWebUI..."
sleep 3
clear
python -c "import os, sys; sys.path.insert(0, os.path.join('$(dirname "${BASH_SOURCE[0]}")', 'LaunchFile')); import app" &

deactivate