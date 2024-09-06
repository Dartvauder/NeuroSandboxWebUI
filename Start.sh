#!/bin/bash

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

source "$CURRENT_DIR/venv/bin/activate"

if [ ! -f Settings.json ]; then
    echo "Settings.json is not found. Please make sure the file exists."
    exit 1
fi

HF_TOKEN=$(grep -oP '"hf_token"\s*:\s*"\K[^"]+' Settings.json)

if [ -z "$HF_TOKEN" ]; then
    echo "HF token is empty or not found in Settings.json. Please add your Hugging Face token to this file."
    exit 1
fi

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
            sleep 120
            xdg-open "http://localhost:7860"
            break
            ;;
        2)
            clear
            python -c "import os, sys; sys.path.insert(0, os.path.join('$(dirname "${BASH_SOURCE[0]}")', 'LaunchFiles')); import appRU" &
            sleep 120
            xdg-open "http://localhost:7860"
            break
            ;;
        3)
            clear
            python -c "import os, sys; sys.path.insert(0, os.path.join('$(dirname "${BASH_SOURCE[0]}")', 'LaunchFiles')); import appAR" &
            sleep 120
            xdg-open "http://localhost:7860"
            break
            ;;
        4)
            clear
            python -c "import os, sys; sys.path.insert(0, os.path.join('$(dirname "${BASH_SOURCE[0]}")', 'LaunchFiles')); import appDE" &
            sleep 120
            xdg-open "http://localhost:7860"
            break
            ;;
        5)
            clear
            python -c "import os, sys; sys.path.insert(0, os.path.join('$(dirname "${BASH_SOURCE[0]}")', 'LaunchFiles')); import appES" &
            sleep 120
            xdg-open "http://localhost:7860"
            break
            ;;
        6)
            clear
            python -c "import os, sys; sys.path.insert(0, os.path.join('$(dirname "${BASH_SOURCE[0]}")', 'LaunchFiles')); import appFR" &
            sleep 120
            xdg-open "http://localhost:7860"
            break
            ;;
        7)
            clear
            python -c "import os, sys; sys.path.insert(0, os.path.join('$(dirname "${BASH_SOURCE[0]}")', 'LaunchFiles')); import appJP" &
            sleep 120
            xdg-open "http://localhost:7860"
            break
            ;;
        8)
            clear
            python -c "import os, sys; sys.path.insert(0, os.path.join('$(dirname "${BASH_SOURCE[0]}")', 'LaunchFiles')); import appZH" &
            sleep 120
            xdg-open "http://localhost:7860"
            break
            ;;
        9)
            clear
            python -c "import os, sys; sys.path.insert(0, os.path.join('$(dirname "${BASH_SOURCE[0]}")', 'LaunchFiles')); import appPT" &
            sleep 120
            xdg-open "http://localhost:7860"
            break
            ;;
        10)
            clear
            python -c "import os, sys; sys.path.insert(0, os.path.join('$(dirname "${BASH_SOURCE[0]}")', 'LaunchFiles')); import appIT" &
            sleep 120
            xdg-open "http://localhost:7860"
            break
            ;;
        11)
            clear
            python -c "import os, sys; sys.path.insert(0, os.path.join('$(dirname "${BASH_SOURCE[0]}")', 'LaunchFiles')); import appHI" &
            sleep 120
            xdg-open "http://localhost:7860"
            break
            ;;
        12)
            clear
            python -c "import os, sys; sys.path.insert(0, os.path.join('$(dirname "${BASH_SOURCE[0]}")', 'LaunchFiles')); import appKO" &
            sleep 120
            xdg-open "http://localhost:7860"
            break
            ;;
        13)
            clear
            python -c "import os, sys; sys.path.insert(0, os.path.join('$(dirname "${BASH_SOURCE[0]}")', 'LaunchFiles')); import appPL" &
            sleep 120
            xdg-open "http://localhost:7860"
            break
            ;;
        14)
            clear
            python -c "import os, sys; sys.path.insert(0, os.path.join('$(dirname "${BASH_SOURCE[0]}")', 'LaunchFiles')); import appTR" &
            sleep 120
            xdg-open "http://localhost:7860"
            break
            ;;
        *)
            echo "Invalid choice, please try again"
            ;;
    esac
done

deactivate
