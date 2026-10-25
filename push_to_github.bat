@echo off
echo ========================================================
echo  GitHub Push Helper for Streamlit Cloud Deployment
echo ========================================================
set /p REPO_URL="Enter your GitHub Repository URL (e.g., https://github.com/username/repo.git): "

if "%REPO_URL%"=="" (
    echo [ERROR] No URL entered! Exiting...
    pause
    exit /b
)

echo.
echo [1/3] Adding remote origin...
git remote remove origin >nul 2>&1
git remote add origin %REPO_URL%

echo [2/3] Setting branch to main...
git branch -M main

echo [3/3] Pushing all files, models, and datasets to GitHub...
git push -u origin main

echo.
echo ========================================================
echo  DONE! Now go to https://share.streamlit.io to deploy!
echo ========================================================
pause
