@echo off
title VisionRadar - Live Production Server
echo ================================================================
echo   VisionRadar: Error-Decomposed Speed Estimation Platform
echo   Starting Backend and Web Interface on http://localhost:8000...
echo ================================================================
echo.

:: Check Python installation
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not found in PATH!
    echo Please install Python 3.10+ or check your environment variables.
    pause
    exit /b 1
)

:: Launch FastAPI server in background
echo Starting VisionRadar Server on http://localhost:8000...
start /b "" python -m uvicorn apps.api.main:app --host 0.0.0.0 --port 8000

:: Wait 3 seconds for server to bind
timeout /t 3 >nul

:: Automatically open browser
echo Opening VisionRadar in your default browser...
start http://localhost:8000

echo.
echo ================================================================
echo   VisionRadar is running!
echo   Local Web App: http://localhost:8000
echo   API Docs:      http://localhost:8000/docs
echo   Health Status: http://localhost:8000/api/v1/health
echo ================================================================
echo.
echo Press CTRL+C or close this window to stop the server.
echo.

:: Keep process alive
python -c "import time; [time.sleep(1) for _ in iter(int, 1)]"
