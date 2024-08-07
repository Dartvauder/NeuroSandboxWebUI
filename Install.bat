@echo off
chcp 65001 > NUL

set CURRENT_DIR=%~dp0
echo Creating virtual environment.../Создание виртуальной среды...
py -m venv "%CURRENT_DIR%venv"
call "%CURRENT_DIR%venv\Scripts\activate.bat"
cls

echo Upgrading pip, setuptools and whell.../Обновление pip, setuptools и wheel...
pip install wheel setuptools pip --upgrade
cls

echo Installing dependencies.../Установка зависимостей...
pip install --no-deps -r "%CURRENT_DIR%requirements.txt"
pip install --no-deps -r "%CURRENT_DIR%requirements-cuda.txt"
pip install --no-deps -r "%CURRENT_DIR%requirements-llama-cpp.txt"
pip install git+https://github.com/tatsy/torchmcubes.git
cls

echo Application has been installed successfully. Run start.bat/Приложение успешно установлено. Запустите start.bat

call "%CURRENT_DIR%venv\Scripts\deactivate.bat"

pause
