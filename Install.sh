#!/bin/bash

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

echo "Creating virtual environment.../Создание виртуальной среды..."
python3 -m venv "$CURRENT_DIR/venv"
source "$CURRENT_DIR/venv/bin/activate"
clear

echo "Upgrading pip, setuptools and whell.../Обновление pip, setuptools и wheel..."
pip install wheel setuptools pip --upgrade
clear

echo "Installing dependencies.../Установка зависимостей..."
pip install --no-deps -r "$CURRENT_DIR/requirements.txt"
pip install --no-deps -r "$CURRENT_DIR/requirements-cuda.txt"
pip install --no-deps -r "$CURRENT_DIR/requirements-llama-cpp.txt"
pip install git+https://github.com/tatsy/torchmcubes.git
clear

echo "Application has been installed successfully. Run start.sh/Приложение успешно установлено. Запустите start.sh"

deactivate

read -p "Press enter to continue/Нажмите enter для продолжения"
