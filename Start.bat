@chcp 65001 > NUL
@echo off

set CURRENT_DIR=%~dp0

call "%CURRENT_DIR%venv\Scripts\activate.bat"

:menu
echo Select a file for launch/Выберите файл для запуска:
echo [English version: 1] appEN.py
echo [Русская версия: 2] appRU.py
echo.

:input
set /p choice=Enter number/Введите число:
if "%choice%"=="1" (
    cls
    start /b py "%CURRENT_DIR%appEN.py"
    timeout /t 30 > NUL
    start http://localhost:7860
    goto end
)

if "%choice%"=="2" (
    cls
    start /b py "%CURRENT_DIR%appRU.py"
    timeout /t 30 > NUL
    start http://localhost:7860
    goto end
)

echo Invalid choice, please try again/Неверный выбор, попробуйте еще раз
goto input

:end
call "%CURRENT_DIR%venv\Scripts\deactivate.bat"
