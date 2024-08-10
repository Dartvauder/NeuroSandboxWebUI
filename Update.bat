@chcp 65001 > NUL
@echo off

git pull

set CURRENT_DIR=%~dp0

call "%CURRENT_DIR%venv\Scripts\activate.bat"

echo Updating dependencies...
python -m pip install --upgrade pip
pip install wheel setuptools
pip install --no-deps -r "%CURRENT_DIR%RequirementsFiles\requirements.txt"
pip install --no-deps -r "%CURRENT_DIR%RequirementsFiles\requirements-cuda.txt"
pip install --no-deps -r "%CURRENT_DIR%RequirementsFiles\requirements-llama-cpp.txt"
pip install git+https://github.com/vork/PyNanoInstantMeshes.git
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/tatsy/torchmcubes.git
cls

echo Application has been updated successfully. Run start.bat

call "%CURRENT_DIR%venv\Scripts\deactivate.bat"

pause
