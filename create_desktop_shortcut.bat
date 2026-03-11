@echo off
chcp 65001 >nul
echo Creating desktop shortcut for AI Olympiad GUI...

REM --- Get paths ---
set "SCRIPT_DIR=%~dp0"
set "DESKTOP=%USERPROFILE%\Desktop"
set "SHORTCUT=%DESKTOP%\AI Olympiad GUI.lnk"
set "TARGET=%SCRIPT_DIR%launch_gui.vbs"

REM --- Create shortcut via PowerShell ---
powershell -NoProfile -Command ^
  "$ws = New-Object -ComObject WScript.Shell; " ^
  "$sc = $ws.CreateShortcut('%SHORTCUT%'); " ^
  "$sc.TargetPath = '%TARGET%'; " ^
  "$sc.WorkingDirectory = '%SCRIPT_DIR%'; " ^
  "$sc.Description = 'AI Olympiad - Emotion Recognition GUI'; " ^
  "$sc.Save()"

if exist "%SHORTCUT%" (
    echo.
    echo [OK] Shortcut created: %SHORTCUT%
    echo You can now launch the GUI from your Desktop!
) else (
    echo [ERROR] Failed to create shortcut.
    echo You can manually copy launch_gui.vbs to your Desktop.
)

pause
