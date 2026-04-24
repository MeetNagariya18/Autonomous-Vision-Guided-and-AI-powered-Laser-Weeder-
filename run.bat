@echo off
title Weed Rover — Command Center
echo.
echo =====================================================
echo   WEED ROVER COMMAND CENTER
echo =====================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.9+
    pause
    exit /b 1
)

:: Install dependencies if needed
echo Checking dependencies...
pip install flask ultralytics opencv-python numpy werkzeug -q

echo.
echo Starting server...
echo Open your browser at: http://127.0.0.1:5000
echo Press Ctrl+C to stop the server.
echo.

python app.py

pause
