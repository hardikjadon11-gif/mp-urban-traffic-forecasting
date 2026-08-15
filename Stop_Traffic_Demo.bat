@echo off
title MP Urban Traffic Forecasting — Stop Demo Server

echo =========================================================================
echo   Stopping MP Urban Traffic Forecasting Demo Server and Tunnel...
echo =========================================================================
echo.

cd /d "%~dp0"

powershell -Command "Get-Process -Name cloudflared -ErrorAction SilentlyContinue | Stop-Process -Force" >nul 2>&1
powershell -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*uvicorn backend.main:app*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }" >nul 2>&1

if exist cloudflared.log del /f /q cloudflared.log >nul 2>&1
if exist backend_output.log del /f /q backend_output.log >nul 2>&1

echo Demo processes stopped cleanly.
ping -n 3 127.0.0.1 >nul
