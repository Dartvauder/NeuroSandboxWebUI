#!/bin/bash

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

source "$CURRENT_DIR/venv/bin/activate"

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
    echo

    read -p "Enter number: " choice

    case $choice in
        1)
            clear
            python "$CURRENT_DIR/LaunchFiles/appEN.py" &
            sleep 120
            xdg-open "http://localhost:7860"
            break
            ;;
        2)
            clear
            python "$CURRENT_DIR/LaunchFiles/appRU.py" &
            sleep 120
            xdg-open "http://localhost:7860"
            break
            ;;
        3)
            clear
            python "$CURRENT_DIR/LaunchFiles/appAR.py" &
            sleep 120
            xdg-open "http://localhost:7860"
            break
            ;;
        4)
            clear
            python "$CURRENT_DIR/LaunchFiles/appDE.py" &
            sleep 120
            xdg-open "http://localhost:7860"
            break
            ;;
        5)
            clear
            python "$CURRENT_DIR/LaunchFiles/appES.py" &
            sleep 120
            xdg-open "http://localhost:7860"
            break
            ;;
        6)
            clear
            python "$CURRENT_DIR/LaunchFiles/appFR.py" &
            sleep 120
            xdg-open "http://localhost:7860"
            break
            ;;
        7)
            clear
            python "$CURRENT_DIR/LaunchFiles/appJP.py" &
            sleep 120
            xdg-open "http://localhost:7860"
            break
            ;;
        8)
            clear
            python "$CURRENT_DIR/LaunchFiles/appZH.py" &
            sleep 120
            xdg-open "http://localhost:7860"
            break
            ;;
        9)
            clear
            python "$CURRENT_DIR/LaunchFiles/appPT.py" &
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
