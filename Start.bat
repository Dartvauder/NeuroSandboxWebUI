@chcp 65001 > NUL
@echo off

set CURRENT_DIR=%~dp0

call "%CURRENT_DIR%venv\Scripts\activate.bat"

echo Attempting to read Settings.json...
if not exist Settings.json (
    echo Settings.json file not found!
    pause
    exit /b
)

for /f "tokens=2 delims=:, " %%a in ('type Settings.json ^| findstr "hf_token"') do set HF_TOKEN=%%~a
echo HF_TOKEN value: "%HF_TOKEN%"
if "%HF_TOKEN%"=="" (
    echo HF token is empty or not found in Settings.json. Please add your Hugging Face token to this file.
    type Settings.json
    pause
    exit /b
)

echo Logging in to Hugging Face...
huggingface-cli login --token %HF_TOKEN% --add-to-git-credential

cls
echo Launching NeuroSandboxWebUI...
start /b py -c "import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__name__), 'LaunchFile')); import app"

call "%CURRENT_DIR%venv\Scripts\deactivate.bat"