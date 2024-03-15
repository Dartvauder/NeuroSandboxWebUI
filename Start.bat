@chcp 65001
@echo off
echo Select a file for launch/Выберите файл для запуска:
echo [English version: 1] appEN.py
echo [Русская версия: 2] appRU.py
echo.
set /p choice=Enter number/Введите число:
if "%choice%"=="1" cmd /k "py appEN.py"
if "%choice%"=="2" cmd /k "py appRU.py"