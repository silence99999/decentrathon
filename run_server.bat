@echo off
set CGO_ENABLED=0
echo Building server...
go build -o server.exe .
if %errorlevel% neq 0 (
    echo Build failed!
    pause
    exit /b 1
)
echo Starting server on http://localhost:8080
echo Press Ctrl+C to stop
server.exe