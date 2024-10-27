@echo off
chcp 65001 > NUL

set CURRENT_DIR=%~dp0

call "%CURRENT_DIR%venv\Scripts\activate.bat"

echo Virtual environment activated
echo.
echo Available options:
echo 1. Install package
echo 2. Uninstall package
echo 3. Exit
echo.

:menu
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    set /p package="Enter package name to install: "
    pip install %package%
    goto menu
)
if "%choice%"=="2" (
    set /p package="Enter package name to uninstall: "
    pip uninstall %package%
    goto menu
)
if "%choice%"=="3" (
    deactivate
    exit
)

echo Invalid choice. Please try again.
goto menu
