@echo off
title MP Urban Traffic Forecasting — 1-Click Production Demo Launcher

echo =========================================================================
echo   Smart Traffic Congestion Forecasting System — Madhya Pradesh
echo   1-Click Production Launcher
echo =========================================================================
echo.

cd /d "%~dp0"

:: 1. Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10+ from python.org and add it to your system PATH.
    echo.
    pause
    exit /b 1
)

:: 2. Check Node / NPM
cmd /c "npm --version" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js / NPM is not installed or not in PATH.
    echo Please install Node.js from nodejs.org to build frontend assets.
    echo.
    pause
    exit /b 1
)

:: 3. Detect cloudflared location
set "CLOUDFLARED_BIN="
where cloudflared >nul 2>&1
if %errorlevel% equ 0 (
    set "CLOUDFLARED_BIN=cloudflared"
) else if exist "C:\Program Files (x86)\cloudflared\cloudflared.exe" (
    set "CLOUDFLARED_BIN=C:\Program Files (x86)\cloudflared\cloudflared.exe"
) else if exist "C:\Program Files\cloudflared\cloudflared.exe" (
    set "CLOUDFLARED_BIN=C:\Program Files\cloudflared\cloudflared.exe"
) else (
    echo [ERROR] Cloudflare tunnel CLI ^(cloudflared^) is not installed.
    echo.
    echo To install cloudflared on Windows:
    echo   Run command: winget install Cloudflare.cloudflared
    echo   Or download installer from: https://github.com/cloudflare/cloudflared/releases
    echo.
    pause
    exit /b 1
)

set "PORT=7860"

:: 4. Clean up old uvicorn or cloudflared instances if running
echo [1/5] Stopping previous demo processes...
powershell -Command "Get-Process -Name uvicorn,cloudflared -ErrorAction SilentlyContinue | Stop-Process -Force" >nul 2>&1

:: 5. Build React Frontend SPA
echo [2/5] Building React production SPA bundle...
cmd /c "npm --prefix frontend run build"
if %errorlevel% neq 0 (
    echo [ERROR] Frontend build failed. Please check frontend dependencies.
    echo.
    pause
    exit /b 1
)

:: 6. Start FastAPI Backend Server
echo [3/5] Starting FastAPI backend on http://0.0.0.0:%PORT% ...
if exist backend_output.log del /f /q backend_output.log >nul 2>&1
start /B python -m uvicorn backend.main:app --host 0.0.0.0 --port %PORT% > backend_output.log 2>&1

echo [4/5] Initializing datasets and ML models...
ping -n 6 127.0.0.1 >nul

:: 7. Start Cloudflare Quick Tunnel
echo [5/5] Launching Cloudflare HTTPS Quick Tunnel...
if exist tunnel.log del /f /q tunnel.log >nul 2>&1
if exist public_url.txt del /f /q public_url.txt >nul 2>&1

start /B "" "%CLOUDFLARED_BIN%" tunnel --url http://localhost:%PORT% > tunnel.log 2>&1

ping -n 6 127.0.0.1 >nul

:: Extract public HTTPS URL from tunnel.log
set "PUBLIC_URL="
powershell -Command "if (Test-Path 'tunnel.log') { $m = Select-String -Path 'tunnel.log' -Pattern 'https://[a-zA-Z0-9-]+\.trycloudflare\.com'; if ($m) { Set-Content -Path 'public_url.txt' -Value $m.Matches[0].Value } }"

if exist public_url.txt set /p PUBLIC_URL=<public_url.txt

echo.
echo =========================================================================
echo   DEMO IS LIVE AND READY FOR COLLEGE PRESENTATION!
echo.
echo   Local URL : http://localhost:%PORT%
if defined PUBLIC_URL (
    echo   PUBLIC URL: %PUBLIC_URL%
    echo.
    echo   Share this PUBLIC URL with any phone or laptop:
    echo   %PUBLIC_URL%
) else (
    echo   [NOTICE] Tunnel is running. Check tunnel.log for your HTTPS link.
)
echo =========================================================================
echo.

if defined PUBLIC_URL start %PUBLIC_URL%

echo Press any key to stop the demo server and tunnel...
pause >nul

call "%~dp0Stop_Traffic_Demo.bat"
