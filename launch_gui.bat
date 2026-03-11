@echo off
chcp 65001 >nul
title AI Olympiad — Emotion Recognition GUI

REM --- Navigate to script directory ---
cd /d "%~dp0"

REM --- Activate virtual environment ---
if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) else (
    echo [ERROR] Virtual environment not found at .venv\
    echo Please run: python -m venv .venv
    pause
    exit /b 1
)

REM --- Launch GUI ---
echo Starting AI Olympiad GUI...
python app.py
if errorlevel 1 (
    echo.
    echo [ERROR] GUI exited with an error.
    pause
)
