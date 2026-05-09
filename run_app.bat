@echo off
setlocal
title ESG Startup Classifier

REM Always run from the folder this .bat lives in
cd /d "%~dp0"

echo ============================================================
echo  ESG Startup Classifier - local launcher
echo ============================================================
echo.

REM ---- 1. Check Python is available -----------------------------------------
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python is not on your PATH.
    echo Install Python 3.10+ from https://www.python.org/downloads/
    echo and make sure to tick "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)

REM ---- 2. Install / refresh dependencies ------------------------------------
echo Checking dependencies (first run may take a minute)...
python -m pip install --quiet --disable-pip-version-check --upgrade pip >nul
python -m pip install --quiet --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install dependencies from requirements.txt.
    echo Try running this command manually to see the full error:
    echo     python -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM ---- 3. Launch Streamlit ---------------------------------------------------
echo.
echo Launching the app... a browser tab will open at http://localhost:8501
echo Close this window or press Ctrl+C to stop the server.
echo.
python -m streamlit run app.py

REM ---- 4. Keep window open on exit so any errors are visible -----------------
echo.
echo App stopped.
pause
