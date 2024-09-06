#!/bin/bash

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

source "$CURRENT_DIR/venv/bin/activate"

# Read token from Settings.json
echo "Attempting to read Settings.json..."
if [ ! -f Settings.json ]; then
    echo "Settings.json file not found!"
    exit 1
fi

echo "Contents of Settings.json:"
cat Settings.json

HF_TOKEN=$(grep -oP '"hf_token"\s*:\s*"\K[^"]+' Settings.json)

echo "HF_TOKEN value: '$HF_TOKEN'"

if [ -z "$HF_TOKEN" ]; then
    echo "HF token is empty or not found in Settings.json. Please add your Hugging Face token to this file."
    exit 1
fi

# Login to Hugging Face
echo "Logging in to Hugging Face..."
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

while true; do
    clear
    echo "Select a file for launch: "
    echo [English version: 1] appEN.py
    echo [Русская версия: 2] appRU.py
    echo [النسخة العربية: 3] appAR.py
    echo [Deutsche Version: 4] appDE.py
    echo [Versión en español: 5] appES.py
    echo [Version française: 6] appFR.py
    echo [日本語版: 7] appJP.py
    echo [中文版: 8] appZH.py
    echo [Versão portuguesa: 9] appPT.py
    echo [Italiano: 10] appIT.py
    echo [हिंदी: 11] appHI.py
    echo [韓國語: 12] appKO.py
    echo [Polski: 13] appPL.py
    echo [Türkçe: 14] appTR.py
    echo

    read -p "Enter number: " choice

    case $choice in
        1)
            clear
            python -c "import os, sys; sys.path.insert(0, os.path.join('$(dirname "${BASH_SOURCE[0]}")', 'LaunchFiles')); import appEN" &

            break
            ;;
        2)
            clear
            python -c "import os, sys; sys.path.insert(0, os.path.join('$(dirname "${BASH_SOURCE[0]}")', 'LaunchFiles')); import appRU" &

            break
            ;;
        3)
            clear
            python -c "import os, sys; sys.path.insert(0, os.path.join('$(dirname "${BASH_SOURCE[0]}")', 'LaunchFiles')); import appAR" &

            break
            ;;
        4)
            clear
            python -c "import os, sys; sys.path.insert(0, os.path.join('$(dirname "${BASH_SOURCE[0]}")', 'LaunchFiles')); import appDE" &

            break
            ;;
        5)
            clear
            python -c "import os, sys; sys.path.insert(0, os.path.join('$(dirname "${BASH_SOURCE[0]}")', 'LaunchFiles')); import appES" &

            break
            ;;
        6)
            clear
            python -c "import os, sys; sys.path.insert(0, os.path.join('$(dirname "${BASH_SOURCE[0]}")', 'LaunchFiles')); import appFR" &

            break
            ;;
        7)
            clear
            python -c "import os, sys; sys.path.insert(0, os.path.join('$(dirname "${BASH_SOURCE[0]}")', 'LaunchFiles')); import appJP" &

            break
            ;;
        8)
            clear
            python -c "import os, sys; sys.path.insert(0, os.path.join('$(dirname "${BASH_SOURCE[0]}")', 'LaunchFiles')); import appZH" &

            break
            ;;
        9)
            clear
            python -c "import os, sys; sys.path.insert(0, os.path.join('$(dirname "${BASH_SOURCE[0]}")', 'LaunchFiles')); import appPT" &

            break
            ;;
        10)
            clear
            python -c "import os, sys; sys.path.insert(0, os.path.join('$(dirname "${BASH_SOURCE[0]}")', 'LaunchFiles')); import appIT" &

            break
            ;;
        11)
            clear
            python -c "import os, sys; sys.path.insert(0, os.path.join('$(dirname "${BASH_SOURCE[0]}")', 'LaunchFiles')); import appHI" &

            break
            ;;
        12)
            clear
            python -c "import os, sys; sys.path.insert(0, os.path.join('$(dirname "${BASH_SOURCE[0]}")', 'LaunchFiles')); import appKO" &

            break
            ;;
        13)
            clear
            python -c "import os, sys; sys.path.insert(0, os.path.join('$(dirname "${BASH_SOURCE[0]}")', 'LaunchFiles')); import appPL" &

            break
            ;;
        14)
            clear
            python -c "import os, sys; sys.path.insert(0, os.path.join('$(dirname "${BASH_SOURCE[0]}")', 'LaunchFiles')); import appTR" &

            break
            ;;
        *)
            echo "Invalid choice, please try again"
            ;;
    esac
done

deactivate
