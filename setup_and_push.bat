@echo off
setlocal
echo ==========================================
echo   Karachi AQI Predictor - Fresh Setup
echo ==========================================

:: Check if git is installed
where git >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Git is not installed. 
    echo Please install it from: https://git-scm.com/downloads
    pause
    exit /b
)

:: Clean up any existing local git info to start totally fresh
if exist .git (
    echo [0/5] Cleaning up old local git data...
    rmdir /s /q .git
)

echo [1/5] Initializing Fresh Git Repository...
git init

echo [2/5] Adding all files...
git add .

echo [3/5] Creating Initial Commit...
git commit -m "Refactored structure: Added app/ and models/ folders"

echo [4/5] Setting main branch...
git branch -M main

echo [5/5] Connecting to GitHub...
:: IMPORTANT: This matches the repo you mentioned in your prompt
git remote add origin https://github.com/MuhammadEhsan02/10Pearls-AQI.git

echo [6/6] Pushing to GitHub...
git push -u origin main --force

echo.
echo ==========================================
echo   DONE! Check your GitHub page now.
echo ==========================================
pause