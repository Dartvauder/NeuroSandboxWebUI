@chcp 65001 > NUL

@echo off

set CURRENT_DIR=%~dp0

call "%CURRENT_DIR%venv\Scripts\activate.bat"

echo Select a file for launch/Выберите файл для запуска:

echo [English version: 1] appEN.py

echo [Русская версия: 2] appRU.py

echo.

set /p choice=Enter number/Введите число:

if "%choice%"=="1" (
    start /b py "%CURRENT_DIR%appEN.py"
    timeout /t 15 > NUL
    start http://localhost:7860
)

if "%choice%"=="2" (
    start /b py "%CURRENT_DIR%appRU.py"
    timeout /t 15 > NUL
    start http://localhost:7860
)

call "%CURRENT_DIR%venv\Scripts\deactivate.bat"
