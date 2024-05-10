@echo off
chcp 65001 > NUL

set CURRENT_DIR=%~dp0

call "%CURRENT_DIR%venv\Scripts\activate.bat"

echo Virtual environment activated/Виртуальное окружение активировано
echo To deactivate the virtual environment, type 'deactivate'/Чтобы деактивировать виртуальное окружение, наберите 'deactivate'

cmd /k