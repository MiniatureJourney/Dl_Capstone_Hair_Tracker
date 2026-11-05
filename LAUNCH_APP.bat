@echo off
title Hair Progress Tracker - Desktop Application
cd /d "%~dp0"
echo ===================================================================
echo   Starting Hair Progress Tracker Desktop GUI...
echo ===================================================================
set NO_ALBUMENTATIONS_UPDATE=1
"C:\Users\Aliasgar\Desktop\DL Capstone\capstone_env\Scripts\python.exe" "app\hair_tracker_app.py" --model "outputs\models\best_model.pth"
if errorlevel 1 (
    echo.
    echo ===================================================================
    echo  Application stopped with an exit code.
    echo ===================================================================
)
pause
