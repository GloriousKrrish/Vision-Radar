@echo off
title VisionRadar — Live Production Server
echo ================================================================
echo   VisionRadar: Error-Decomposed Speed Estimation Platform
echo   Launching Frontend, Backend, and Live Global Edge Tunnel...
echo ================================================================
echo.

:: Ensure python and dependencies
start /b "" python -m uvicorn apps.api.main:app --host 0.0.0.0 --port 8000
timeout /t 3 >nul

echo Server started on port 8000. Launching Cloudflare public tunnel...
.\cloudflared.exe tunnel --url http://localhost:8000
