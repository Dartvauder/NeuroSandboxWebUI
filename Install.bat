@echo off
chcp 65001 > NUL

set CURRENT_DIR=%~dp0
echo Creating virtual environment...
py -m venv "%CURRENT_DIR%venv"
call "%CURRENT_DIR%venv\Scripts\activate.bat"
cls

echo Upgrading pip, setuptools and wheel...
python -m pip install --upgrade pip
pip install wheel setuptools
timeout /t 3 /nobreak >nul
cls

echo Installing dependencies...
if not exist "%CURRENT_DIR%logs" mkdir "%CURRENT_DIR%logs"
set ERROR_LOG="%CURRENT_DIR%logs\installation_errors.log"
type nul > %ERROR_LOG%

set BUILD_CUDA_EXT=1
set INSTALL_KERNELS=1

pip install --no-deps -r "%CURRENT_DIR%RequirementsFiles\requirements.txt" 2>> %ERROR_LOG%
pip install --no-deps -r "%CURRENT_DIR%RequirementsFiles\requirements-cuda.txt" 2>> %ERROR_LOG%
pip install --no-deps -r "%CURRENT_DIR%RequirementsFiles\requirements-llama-cpp.txt" 2>> %ERROR_LOG%
pip install --no-deps -r "%CURRENT_DIR%RequirementsFiles\requirements-stable-diffusion-cpp.txt" 2>> %ERROR_LOG%
pip install --no-build-isolation -e git+https://github.com/PanQiWei/AutoGPTQ.git#egg=auto_gptq 2>> "$ERROR_LOG"
pip install https://huggingface.co/madbuda/triton-windows-builds/resolve/main/triton-2.1.0-cp310-cp310-win_amd64.whl 2>> "$ERROR_LOG"
pip install --no-build-isolation -e git+https://github.com/casper-hansen/AutoAWQ.git#egg=autoawq 2>> "$ERROR_LOG"
pip install --no-build-isolation -e git+https://github.com/turboderp/exllamav2.git#egg=exllamav2 2>> "$ERROR_LOG"
pip install git+https://github.com/tencent-ailab/IP-Adapter.git 2>> %ERROR_LOG%
pip install git+https://github.com/vork/PyNanoInstantMeshes.git 2>> %ERROR_LOG%
pip install git+https://github.com/openai/CLIP.git 2>> %ERROR_LOG%
timeout /t 3 /nobreak >nul
cls

echo Post-installing patches...
python "%CURRENT_DIR%RequirementsFiles\post_install.py"
timeout /t 3 /nobreak >nul
cls

echo Checking for installation errors...
findstr /C:"error" %ERROR_LOG% >nul
if %ERRORLEVEL% equ 0 (
    echo Some packages failed to install. Please check %ERROR_LOG% for details.
) else (
    echo Installation completed successfully.
)
timeout /t 5 /nobreak >nul
cls

echo Application installation process completed. Run start.bat to launch the application.

call "%CURRENT_DIR%venv\Scripts\deactivate.bat"

pause