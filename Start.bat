@chcp 65001 > NUL
@echo off

set CURRENT_DIR=%~dp0

call "%CURRENT_DIR%venv\Scripts\activate.bat"

:menu
echo Select a file for launch:
echo [English version: 1] appEN.py
echo [Русская версия: 2] appRU.py
echo [النسخة العربية: 3] appAR.py
echo [Deutsche Version: 4] appDE.py
echo [Versión en español: 5] appES.py
echo [Version française: 6] appFR.py
echo [日本語版: 7] appJP.py
echo [中文版: 8] appZH.py
echo [Versão portuguesa: 9] appPT.py
echo [Italiano: 10] appIT.py
echo [हिंदी: 11] appHI.py
echo.

:input
set /p choice=Enter number:
if "%choice%"=="1" (
    cls
    start /b py "%CURRENT_DIR%LaunchFiles\appEN.py"
    timeout /t 120 > NUL
    start http://localhost:7860
    goto end
)

if "%choice%"=="2" (
    cls
    start /b py "%CURRENT_DIR%LaunchFiles\appRU.py"
    timeout /t 120 > NUL
    start http://localhost:7860
    goto end
)

if "%choice%"=="3" (
    cls
    start /b py "%CURRENT_DIR%LaunchFiles\appAR.py"
    timeout /t 120 > NUL
    start http://localhost:7860
    goto end
)

if "%choice%"=="4" (
    cls
    start /b py "%CURRENT_DIR%LaunchFiles\appDE.py"
    timeout /t 120 > NUL
    start http://localhost:7860
    goto end
)

if "%choice%"=="5" (
    cls
    start /b py "%CURRENT_DIR%LaunchFiles\appES.py"
    timeout /t 120 > NUL
    start http://localhost:7860
    goto end
)

if "%choice%"=="6" (
    cls
    start /b py "%CURRENT_DIR%LaunchFiles\appFR.py"
    timeout /t 120 > NUL
    start http://localhost:7860
    goto end
)

if "%choice%"=="7" (
    cls
    start /b py "%CURRENT_DIR%LaunchFiles\appJP.py"
    timeout /t 120 > NUL
    start http://localhost:7860
    goto end
)

if "%choice%"=="8" (
    cls
    start /b py "%CURRENT_DIR%LaunchFiles\appZH.py"
    timeout /t 120 > NUL
    start http://localhost:7860
    goto end
)

if "%choice%"=="9" (
    cls
    start /b py "%CURRENT_DIR%LaunchFiles\appPT.py"
    timeout /t 120 > NUL
    start http://localhost:7860
    goto end
)

if "%choice%"=="10" (
    cls
    start /b py "%CURRENT_DIR%LaunchFiles\appIT.py"
    timeout /t 120 > NUL
    start http://localhost:7860
    goto end
)

if "%choice%"=="11" (
    cls
    start /b py "%CURRENT_DIR%LaunchFiles\appHI.py"
    timeout /t 120 > NUL
    start http://localhost:7860
    goto end
)

echo Invalid choice, please try again
goto input

:end
call "%CURRENT_DIR%venv\Scripts\deactivate.bat"
