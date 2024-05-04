@chcp 65001 > NUL
@echo off

git pull

set CURRENT_DIR=%~dp0

call "%CURRENT_DIR%venv\Scripts\activate.bat"

echo Updating dependencies.../Обновление зависимостей...
pip install --no-deps -r "%CURRENT_DIR%requirements.txt"
pip install --no-deps -r "%CURRENT_DIR%requirements-cuda.txt"
pip install --no-deps -r "%CURRENT_DIR%requirements-llama-cpp.txt"
cls

echo Application has been updated successfully. Run start.bat/Приложение успешно обновлено. Запустите start.bat

call "%CURRENT_DIR%venv\Scripts\deactivate.bat"

pause
