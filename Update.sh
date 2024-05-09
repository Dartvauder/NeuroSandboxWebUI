#!/bin/bash

git pull

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

source "$CURRENT_DIR/venv/bin/activate"

echo "Updating dependencies.../Обновление зависимостей..."
pip install --no-deps -r "$CURRENT_DIR/requirements.txt"
pip install --no-deps -r "$CURRENT_DIR/requirements-cuda.txt"
pip install --no-deps -r "$CURRENT_DIR/requirements-llama-cpp.txt"
pip install git+https://github.com/tatsy/torchmcubes.git
pip install flash-attn==2.5.8 --no-build-isolation
clear

echo "Application has been updated successfully. Run start.sh/Приложение успешно обновлено. Запустите start.sh"

deactivate

read -p "Press enter to continue/Нажмите enter для продолжения"
