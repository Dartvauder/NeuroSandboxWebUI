@echo off
chcp 65001 > NUL

set CURRENT_DIR=%~dp0
echo Creating virtual environment...
py -m venv "%CURRENT_DIR%venv"
call "%CURRENT_DIR%venv\Scripts\activate.bat"
cls

echo Upgrading pip, setuptools and whell...
python -m pip install --upgrade pip
pip install wheel setuptools
cls

echo Installing dependencies...
pip install --no-deps -r "%CURRENT_DIR%RequirementsFiles\requirements.txt"
pip install --no-deps -r "%CURRENT_DIR%RequirementsFiles\requirements-cuda.txt"
pip install --no-deps -r "%CURRENT_DIR%RequirementsFiles\requirements-llama-cpp.txt"
pip install git+https://github.com/tencent-ailab/IP-Adapter.git
pip install git+https://github.com/vork/PyNanoInstantMeshes.git
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/tatsy/torchmcubes.git
cls

echo Post-installing patches...
python "%CURRENT_DIR%RequirementsFiles\post_install.py"
cls

echo Application has been installed successfully. Run start.bat

call "%CURRENT_DIR%venv\Scripts\deactivate.bat"

pause
