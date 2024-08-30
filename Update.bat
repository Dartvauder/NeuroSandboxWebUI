@chcp 65001 > NUL
@echo off

git pull
timeout /t 3 /nobreak >nul
cls

set CURRENT_DIR=%~dp0

call "%CURRENT_DIR%venv\Scripts\activate.bat"

echo Updating dependencies...
if not exist "%CURRENT_DIR%logs" mkdir "%CURRENT_DIR%logs"
set ERROR_LOG="%CURRENT_DIR%logs\update_errors.log"
type nul > %ERROR_LOG%

python -m pip install --upgrade pip
pip install wheel setuptools
pip install --no-deps -r "%CURRENT_DIR%RequirementsFiles\requirements.txt" 2>> %ERROR_LOG%
pip install --no-deps -r "%CURRENT_DIR%RequirementsFiles\requirements-cuda.txt" 2>> %ERROR_LOG%
pip install --no-deps -r "%CURRENT_DIR%RequirementsFiles\requirements-llama-cpp.txt" 2>> %ERROR_LOG%
pip install git+https://github.com/tencent-ailab/IP-Adapter.git 2>> %ERROR_LOG%
pip install git+https://github.com/vork/PyNanoInstantMeshes.git 2>> %ERROR_LOG%
pip install git+https://github.com/openai/CLIP.git 2>> %ERROR_LOG%
pip install git+https://github.com/tatsy/torchmcubes.git 2>> %ERROR_LOG%
timeout /t 3 /nobreak >nul
cls

echo Post-installing patches...
python "%CURRENT_DIR%RequirementsFiles\post_install.py"
timeout /t 3 /nobreak >nul
cls

echo Checking for update errors...
findstr /V /C:"ERROR: ERROR" %ERROR_LOG% > %ERROR_LOG%.tmp
move /Y %ERROR_LOG%.tmp %ERROR_LOG% > nul

if %ERRORLEVEL% neq 0 (
    echo Some packages failed to update. Please check %ERROR_LOG% for details.
    echo You can try to update these packages manually by running:
    echo call "%CURRENT_DIR%venv.bat"
    echo and then use pip to install the missing packages.
) else (
    echo All packages updated successfully.
    del %ERROR_LOG%
)
timeout /t 3 /nobreak >nul
cls

echo Application update process completed. Run start.bat to launch the application.

call "%CURRENT_DIR%venv\Scripts\deactivate.bat"

pause