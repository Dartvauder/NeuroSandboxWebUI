@chcp 65001 > NUL
@echo off

git pull

set CURRENT_DIR=%~dp0

call "%CURRENT_DIR%venv\Scripts\activate.bat"

pause
