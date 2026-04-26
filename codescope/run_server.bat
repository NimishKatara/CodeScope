@echo off
REM CodeScope Server Launcher
REM Sets up the environment and starts the Flask server

setlocal enabledelayedexpansion

REM Set the Google API Key
set GOOGLE_API_KEY=AIzaSyDwBN7NRfW8Vc6Vbiq2tHjpLl29YPjGUfw

REM Navigate to backend/codebaseagent directory
cd /d "%~dp0backend\codebaseagent"

echo.
echo ================================================
echo CodeScope API Server
echo ================================================
echo API Key: %GOOGLE_API_KEY%
echo Starting server on http://localhost:5000
echo ================================================
echo.

REM Run the server
python server.py

pause
