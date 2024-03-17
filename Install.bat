@echo off
chcp 65001 > NUL

set CURRENT_DIR=%~dp0
py -m venv "%CURRENT_DIR%venv"
call "%CURRENT_DIR%venv\Scripts\activate.bat"

:choice
set /p choice="Выберите версию для установки/choose a version for install: 1) requirementsGPU.txt 2) requirementsCPU.txt: "

if "%choice%"=="1" (
    echo The application is installing, please wait/Приложение устанавливается, пожалуйста подождите.
    pip install -r "%CURRENT_DIR%requirementsGPU.txt"
) else if "%choice%"=="2" (
    echo The application is installing, please wait/Приложение устанавливается, пожалуйста подождите.
    pip install -r "%CURRENT_DIR%requirementsCPU.txt"
) else (
    echo Wronge choose. Try again/Неверный выбор. Попробуйте еще раз.
    goto choice
)

echo The application has been installed successfully. Run start.bat/Приложение успешно установлено. Запустите start.bat.
pause
