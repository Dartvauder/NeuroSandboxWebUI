#!/bin/bash

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${CURRENT_DIR}/venv/bin/activate"

echo "Virtual environment activated/Виртуальное окружение активировано"
echo "To deactivate the virtual environment, type 'deactivate'/Чтобы деактивировать виртуальное окружение, наберите 'deactivate'."

bash