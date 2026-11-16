@echo off
set "PYTHON=C:\Python315\python.exe"
if not exist "%PYTHON%" (
    echo Python was not found at %PYTHON%.
    pause
    exit /b 1
)
cd /d "%~dp0"
"%PYTHON%" -m streamlit run "%~dp0anova_app.py"
echo.
pause
