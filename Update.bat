@chcp 65001 > NUL
@echo off

git pull
timeout /t 3 /nobreak >nul
cls

set CURRENT_DIR=%~dp0

call "%CURRENT_DIR%venv\Scripts\activate.bat"

echo Setting up local pip cache...
if not exist "%CURRENT_DIR%pip_cache" mkdir "%CURRENT_DIR%pip_cache"
set PIP_CACHE_DIR=%CURRENT_DIR%pip_cache

echo Updating dependencies...
if not exist "%CURRENT_DIR%logs" mkdir "%CURRENT_DIR%logs"
set ERROR_LOG="%CURRENT_DIR%logs\update_errors.log"
type nul > %ERROR_LOG%

set BUILD_CUDA_EXT=1
set INSTALL_KERNELS=1

python -m pip install --upgrade pip
pip install wheel setuptools
pip install --no-deps -r "%CURRENT_DIR%RequirementsFiles\requirements.txt" 2>> %ERROR_LOG%
pip install --no-deps -r "%CURRENT_DIR%RequirementsFiles\requirements-cuda.txt" 2>> %ERROR_LOG%
pip install --no-deps -r "%CURRENT_DIR%RequirementsFiles\requirements-llama-cpp.txt" 2>> %ERROR_LOG%
pip install --no-deps -r "%CURRENT_DIR%RequirementsFiles\requirements-stable-diffusion-cpp.txt" 2>> %ERROR_LOG%
pip install --no-build-isolation -e git+https://github.com/PanQiWei/AutoGPTQ.git#egg=auto_gptq 2>> %ERROR_LOG%
pip install https://huggingface.co/madbuda/triton-windows-builds/resolve/main/triton-2.1.0-cp310-cp310-win_amd64.whl 2>> %ERROR_LOG%
pip install --no-build-isolation -e git+https://github.com/casper-hansen/AutoAWQ.git#egg=autoawq 2>> %ERROR_LOG%
pip install --no-build-isolation -e git+https://github.com/turboderp/exllamav2.git#egg=exllamav2 2>> %ERROR_LOG%
pip install git+https://github.com/tencent-ailab/IP-Adapter.git 2>> %ERROR_LOG%
pip install git+https://github.com/vork/PyNanoInstantMeshes.git 2>> %ERROR_LOG%
pip install git+https://github.com/openai/CLIP.git 2>> %ERROR_LOG%
pip install git+https://github.com/xhinker/sd_embed.git@main 2>> %ERROR_LOG%
timeout /t 3 /nobreak >nul
cls

echo Post-installing patches...
python "%CURRENT_DIR%RequirementsFiles\post_install.py"
timeout /t 3 /nobreak >nul
cls

echo Checking for update errors...
findstr /C:"error" %ERROR_LOG% >nul
if %ERRORLEVEL% equ 0 (
    echo Some packages failed to install. Please check %ERROR_LOG% for details.
) else (
    echo Installation completed successfully.
)
timeout /t 5 /nobreak >nul
cls

echo Application update process completed. Run start.bat to launch the application.

call "%CURRENT_DIR%venv\Scripts\deactivate.bat"

pause