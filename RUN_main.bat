@echo off
cd /d "%~dp0"
call ".venv\Scripts\activate.bat"
python main.py
echo.
echo Finished. Press any key to close.
pause >nul
