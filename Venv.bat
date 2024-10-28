@echo off
chcp 65001 > NUL

set CURRENT_DIR=%~dp0

call "%CURRENT_DIR%venv\Scripts\activate.bat"

echo Virtual environment activated
echo.
echo Available options:
echo 1. Install package
echo 2. Uninstall package
echo 3. Upgrade package
echo 4. List installed packages
echo 5. Show package details
echo 6. Check dependencies
echo 7. Debug information
echo 8. Exit
echo.

:menu
set /p choice="Enter your choice (1-8): "

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
    set /p package="Enter package name to upgrade: "
    pip install --upgrade %package%
    goto menu
)
if "%choice%"=="4" (
    pip list
    echo.
    goto menu
)
if "%choice%"=="5" (
    set /p package="Enter package name to show details: "
    pip show %package%
    echo.
    goto menu
)
if "%choice%"=="6" (
    pip check
    echo.
    goto menu
)
if "%choice%"=="7" (
    pip debug --verbose
    echo.
    goto menu
)
if "%choice%"=="8" (
    deactivate
    exit
)

echo Invalid choice. Please try again.
goto menu