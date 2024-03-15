@chcp 65001
@echo off

set CURRENT_DIR=%~dp0

py -m venv "%CURRENT_DIR%venv"

call "%CURRENT_DIR%venv\Scripts\activate.bat"

echo Python virtual environment activated/Виртуальное окружение Python активировано.

pip install -r "%CURRENT_DIR%requirements.txt"

echo Packages from requirements.txt were successfully installed/Пакеты из requirements.txt успешно установлены.

pause
