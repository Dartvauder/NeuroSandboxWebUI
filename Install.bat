@chcp 65001 > NUL
@echo off

set CURRENT_DIR=%~dp0

py -m venv "%CURRENT_DIR%venv"

call "%CURRENT_DIR%venv\Scripts\activate.bat"

echo The application is installing, please wait/Приложение устанавливается, пожалуйста подождите.

pip install -r "%CURRENT_DIR%requirements.txt"

echo The application has been installed successfully. Run start.bat/Приложение успешно установлено. Запустите start.bat.

pause
