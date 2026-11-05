@echo off
title Hair Progress Tracker - Live Web Deployment
cd /d "%~dp0"
echo ===================================================================
echo   Starting Hair Progress Tracker Live Web Deployment...
echo ===================================================================

:: Pre-launch cleanup of any lingering tunnels
taskkill /F /IM cloudflared.exe >nul 2>&1

set NO_ALBUMENTATIONS_UPDATE=1
"C:\Users\Aliasgar\Desktop\DL Capstone\capstone_env\Scripts\python.exe" run_web_app.py
if errorlevel 1 (
    echo.
    echo ===================================================================
    echo  Server stopped with an exit code.
    echo  Check tunnel.log in this folder if there was a network issue.
    echo ===================================================================
)
pause
