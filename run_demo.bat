@echo off
chcp 65001 >nul 2>nul
cd /d "%~dp0"
title AI Olympiad Demo
echo ============================================
echo   AI Olympiad — Demo Launcher
echo ============================================
echo.

REM --- Find Python venv ---
set VENV=.venv
if not exist %VENV%\Scripts\python.exe (
    set VENV=..\..\..\..venv
)
if not exist %VENV%\Scripts\python.exe (
    echo [1/4] Creating virtual environment...
    set VENV=.venv
    python -m venv %VENV%
)

set PY=%VENV%\Scripts\python.exe
set PIP=%VENV%\Scripts\pip.exe

REM --- Install wheels offline ---
echo [2/4] Installing dependencies from wheels...
%PIP% install --no-index --find-links tools\wheels torch torchvision torchaudio 2>nul
%PIP% install --no-index --find-links tools\wheels opencv-python PyQt5 numpy scikit-learn Pillow matplotlib 2>nul
%PIP% install --no-index --find-links tools\wheels -r requirements.txt 2>nul
echo.

REM --- Menu ---
:menu
echo Choose an action:
echo   1 — Train emotion model      (train.py)
echo   2 — Camera emotion inference  (infer.py)
echo   3 — Full GUI application      (app.py)
echo   4 — Exit
echo.
set /p choice="> "

if "%choice%"=="1" (
    echo Running training...
    %PY% train.py --model mobilenet --epochs 40 --batch 64 --img_size 128 --lr 0.0003
    goto menu
)
if "%choice%"=="2" (
    echo Running camera inference...
    %PY% infer.py --source 0 --img_size 128
    goto menu
)
if "%choice%"=="3" (
    echo Running GUI application...
    %PY% app.py
    goto menu
)
if "%choice%"=="4" (
    echo Bye!
    exit /b 0
)
echo Invalid choice.
goto menu
