# CodeScope Server Launcher (PowerShell)
# Sets API key and starts the Flask server

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "   CodeScope API Server (Groq)" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Set the Groq API Key (replace with your actual key)
$env:GROQ_API_KEY = 'API_KEY'

Write-Host "✓ Groq API Key set" -ForegroundColor Green
Write-Host "✓ Starting Flask server on http://localhost:5000" -ForegroundColor Green
Write-Host ""

# Navigate to backend/codebaseagent
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location "$scriptPath\backend\codebaseagent"

# Run the server
python server.py
