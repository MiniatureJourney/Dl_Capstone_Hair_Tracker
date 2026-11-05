@echo off
title Stop Hair Progress Tracker Services
cd /d "%~dp0"
echo ===================================================================
echo   Stopping All Hair Progress Tracker Services...
echo ===================================================================

echo [*] Terminating any Cloudflare tunnels...
taskkill /F /IM cloudflared.exe >nul 2>&1

echo [*] Freeing port 7860...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :7860 ^| findstr LISTENING') do (
    taskkill /F /T /PID %%a >nul 2>&1
)

echo [*] Done! All services stopped and port 7860 is free.
echo ===================================================================
timeout /t 2 >nul
