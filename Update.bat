@chcp 65001 > NUL
@echo off

git pull
timeout /t 3 /nobreak >nul
cls

set CURRENT_DIR=%~dp0

if exist "%CURRENT_DIR%install_config.txt" (
    for /f "tokens=2 delims==" %%a in ('type "%CURRENT_DIR%install_config.txt" ^| findstr "INSTALL_TYPE"') do set INSTALL_TYPE=%%a
) else (
    echo Installation configuration not found. Please run install.bat first.
    pause
    exit /b 1
)

call "%CURRENT_DIR%venv\Scripts\activate.bat"

echo Setting up local pip cache...
if not exist "%CURRENT_DIR%pip_cache" mkdir "%CURRENT_DIR%pip_cache"
set PIP_CACHE_DIR=%CURRENT_DIR%pip_cache

echo Updating dependencies for %INSTALL_TYPE% version...
if not exist "%CURRENT_DIR%logs" mkdir "%CURRENT_DIR%logs"
set ERROR_LOG="%CURRENT_DIR%logs\update_errors.log"
type nul > %ERROR_LOG%

if "%INSTALL_TYPE%"=="CPU" (
    set BUILD_CUDA_EXT=0
    set INSTALL_KERNELS=0
) else (
    set BUILD_CUDA_EXT=1
    set INSTALL_KERNELS=1
)

python -m pip install --upgrade pip setuptools
pip install wheel

if "%INSTALL_TYPE%"=="CPU" (
    pip install --no-deps -r "%CURRENT_DIR%RequirementsFiles\requirements-СPU.txt" 2>> %ERROR_LOG%
    pip install --no-deps -r "%CURRENT_DIR%RequirementsFiles\requirements-cuda-CPU.txt" 2>> %ERROR_LOG%
    pip install --no-deps -r "%CURRENT_DIR%RequirementsFiles\requirements-llama-cpp-CPU.txt" 2>> %ERROR_LOG%
    pip install --no-deps -r "%CURRENT_DIR%RequirementsFiles\requirements-stable-diffusion-cpp-CPU.txt" 2>> %ERROR_LOG%
) else (
    pip install --no-deps -r "%CURRENT_DIR%RequirementsFiles\requirements.txt" 2>> %ERROR_LOG%
    pip install --no-deps -r "%CURRENT_DIR%RequirementsFiles\requirements-cuda.txt" 2>> %ERROR_LOG%
    pip install --no-deps -r "%CURRENT_DIR%RequirementsFiles\requirements-llama-cpp.txt" 2>> %ERROR_LOG%
    pip install --no-deps -r "%CURRENT_DIR%RequirementsFiles\requirements-stable-diffusion-cpp.txt" 2>> %ERROR_LOG%
)

pip install --no-deps WinTmp==1.1.0
pip install https://huggingface.co/madbuda/triton-windows-builds/resolve/main/triton-3.0.0-cp310-cp310-win_amd64.whl 2>> %ERROR_LOG%
pip install --no-build-isolation -e git+https://github.com/PanQiWei/AutoGPTQ.git#egg=auto_gptq@v0.7.1 2>> %ERROR_LOG%
pip install --no-build-isolation -e git+https://github.com/casper-hansen/AutoAWQ.git#egg=autoawq@v0.2.6 2>> %ERROR_LOG%
pip install --no-build-isolation -e git+https://github.com/turboderp/exllamav2.git#egg=exllamav2@v0.2.3 2>> %ERROR_LOG%
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