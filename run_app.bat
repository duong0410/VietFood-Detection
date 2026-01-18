@echo off
echo ========================================
echo VietFood Detection Application
echo ========================================
echo.

REM Activate virtual environment if exists
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Run the application
echo Starting application...
python main.py

pause
