@echo off
chcp 65001 > NUL

set CURRENT_DIR=%~dp0
echo Creating virtual environment.../Создание виртуальной среды...
py -m venv "%CURRENT_DIR%venv"
call "%CURRENT_DIR%venv\Scripts\activate.bat"
cls

echo Upgrading pip and setuptools.../Обновление pip и setuptools...
py -m pip install --upgrade pip setuptools
cls

echo Installing dependencies.../Установка зависимостей...
pip install --no-deps -r "%CURRENT_DIR%requirements.txt"
pip install --no-deps -r "%CURRENT_DIR%requirements-cuda.txt"
pip install --no-deps -r "%CURRENT_DIR%requirements-llama-cpp.txt"
cls

echo Application has been installed successfully. Run start.bat/Приложение успешно установлено. Запустите start.bat

call "%CURRENT_DIR%venv\Scripts\deactivate.bat"

pause
