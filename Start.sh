#!/bin/bash

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

source "$CURRENT_DIR/venv/bin/activate"

while true; do
    clear
    echo "Select a file for launch/Выберите файл для запуска:"
    echo "[English version: 1] appEN.py"
    echo "[Русская версия: 2] appRU.py"
    echo

    read -p "Enter number/Введите число: " choice

    case $choice in
        1)
            clear
            python "$CURRENT_DIR/appEN.py" &
            sleep 120
            xdg-open "http://localhost:7860"
            break
            ;;
        2)
            clear
            python "$CURRENT_DIR/appRU.py" &
            sleep 120
            xdg-open "http://localhost:7860"
            break
            ;;
        *)
            echo "Invalid choice, please try again/Неверный выбор, попробуйте еще раз"
            ;;
    esac
done

deactivate
