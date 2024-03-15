@chcp 65001
@echo off

set CURRENT_DIR=%~dp0

call "%CURRENT_DIR%venv\Scripts\activate.bat"

echo Select a file for launch/Выберите файл для запуска:
echo [English version: 1] appEN.py
echo [Русская версия: 2] appRU.py
echo.
set /p choice=Enter number/Введите число:

if "%choice%"=="1" py "%CURRENT_DIR%appEN.py"
if "%choice%"=="2" py "%CURRENT_DIR%appRU.py"

call "%CURRENT_DIR%venv\Scripts\deactivate.bat"
