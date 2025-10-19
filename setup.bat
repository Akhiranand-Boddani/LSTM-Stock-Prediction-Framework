@echo off
REM LSTM Stock Prediction - Windows Setup Script
REM This script automates the initial setup and GitHub upload process

echo ================================================
echo LSTM Stock Prediction - Setup and GitHub Upload
echo ================================================
echo.

REM Check if git is installed
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Git is not installed!
    echo Please install Git from: https://git-scm.com/downloads
    pause
    exit /b 1
)
echo [OK] Git is installed

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed!
    echo Please install Python 3.8+ from: https://www.python.org/downloads/
    pause
    exit /b 1
)
echo [OK] Python is installed
echo.

echo ================================================
echo Step 1: Git Initialization
echo ================================================
echo.

REM Initialize git repository
if exist .git (
    echo [INFO] Git repository already initialized
) else (
    echo [INFO] Initializing git repository...
    git init
    echo [OK] Git repository initialized
)
echo.

echo ================================================
echo Step 2: Configure Git User
echo ================================================
echo.

REM Check if git user is configured
for /f "tokens=*" %%i in ('git config user.name') do set GIT_NAME=%%i
for /f "tokens=*" %%i in ('git config user.email') do set GIT_EMAIL=%%i

if "%GIT_NAME%"=="" (
    set /p USER_NAME="Enter your name: "
    git config user.name "%USER_NAME%"
    echo [OK] Git user name set
) else (
    echo [INFO] Git user name: %GIT_NAME%
)

if "%GIT_EMAIL%"=="" (
    set /p USER_EMAIL="Enter your email: "
    git config user.email "%USER_EMAIL%"
    echo [OK] Git email set
) else (
    echo [INFO] Git email: %GIT_EMAIL%
)
echo.

echo ================================================
echo Step 3: Stage Files
echo ================================================
echo.

echo [INFO] Staging all files...
git add .
echo [OK] Files staged
echo.

echo Files to be committed:
git status --short
echo.

set /p CONTINUE="Do you want to continue? (Y/N): "
if /i not "%CONTINUE%"=="Y" (
    echo [ERROR] Setup cancelled
    pause
    exit /b 1
)

echo.
echo ================================================
echo Step 4: Create Initial Commit
echo ================================================
echo.

echo [INFO] Creating initial commit...
git commit -m "Initial commit: LSTM Stock Prediction Framework"
echo [OK] Initial commit created
echo.

echo ================================================
echo Step 5: GitHub Repository Setup
echo ================================================
echo.

echo [INFO] Please create a repository on GitHub:
echo   1. Go to https://github.com/new
echo   2. Repository name: LSTM-Stock-Prediction-Framework
echo   3. Description: LSTM-based stock price prediction with Streamlit
echo   4. Visibility: Public (recommended)
echo   5. DO NOT initialize with README, .gitignore, or license
echo   6. Click 'Create repository'
echo.

set /p GITHUB_USERNAME="Enter your GitHub username: "

if "%GITHUB_USERNAME%"=="" (
    echo [ERROR] GitHub username is required
    pause
    exit /b 1
)

set REPO_URL=https://github.com/%GITHUB_USERNAME%/LSTM-Stock-Prediction-Framework.git
echo.
echo [INFO] Repository URL: %REPO_URL%
echo.

REM Check if remote already exists
git remote | findstr "^origin$" >nul 2>&1
if %errorlevel% equ 0 (
    echo [INFO] Remote 'origin' already exists. Updating URL...
    git remote set-url origin %REPO_URL%
) else (
    echo [INFO] Adding remote repository...
    git remote add origin %REPO_URL%
)
echo [OK] Remote repository configured
echo.

echo ================================================
echo Step 6: Push to GitHub
echo ================================================
echo.

echo [INFO] Renaming branch to 'main'...
git branch -M main
echo [OK] Branch renamed to 'main'
echo.

echo [INFO] Pushing to GitHub...
echo.
echo [NOTE] You may be prompted for GitHub credentials
echo [NOTE] If using 2FA, use a Personal Access Token instead of password
echo [NOTE] Generate token at: https://github.com/settings/tokens
echo.

git push -u origin main

if %errorlevel% equ 0 (
    echo.
    echo ================================================
    echo SUCCESS! Setup Complete!
    echo ================================================
    echo.
    echo [OK] Your repository is now available at:
    echo   https://github.com/%GITHUB_USERNAME%/LSTM-Stock-Prediction-Framework
    echo.
    echo Next steps:
    echo   1. Visit your repository on GitHub
    echo   2. Deploy to Streamlit Cloud: https://share.streamlit.io
    echo   3. See DEPLOYMENT.md for detailed instructions
    echo.
    
    set /p OPEN_BROWSER="Would you like to open your repository in the browser? (Y/N): "
    if /i "%OPEN_BROWSER%"=="Y" (
        start https://github.com/%GITHUB_USERNAME%/LSTM-Stock-Prediction-Framework
    )
) else (
    echo.
    echo [ERROR] Failed to push to GitHub
    echo.
    echo Common issues and solutions:
    echo   1. Authentication failed:
    echo      - Use Personal Access Token instead of password
    echo      - Generate at: https://github.com/settings/tokens
    echo.
    echo   2. Repository doesn't exist:
    echo      - Make sure you created the repository on GitHub first
    echo      - Check the repository name matches: LSTM-Stock-Prediction-Framework
    echo.
    echo   3. Permission denied:
    echo      - Verify your GitHub username is correct
    echo      - Check repository visibility settings
    echo.
    echo Manual push command:
    echo   git push -u origin main
    echo.
)

echo.
pause