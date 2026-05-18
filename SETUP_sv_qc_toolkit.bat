@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
title sv_qc_toolkit setup

echo.
echo ============================================================
echo  sv_qc_toolkit setup
echo ============================================================
echo.

cd /d "%~dp0"
echo Project folder:
echo %CD%
echo.

REM ------------------------------------------------------------
REM Choose Python. Prefer Python 3.12/3.11 if installed,
REM otherwise use default Python.
REM ------------------------------------------------------------
set "PYTHON_CMD="

py -3.12 --version >nul 2>nul
if not errorlevel 1 set "PYTHON_CMD=py -3.12"

if not defined PYTHON_CMD (
    py -3.11 --version >nul 2>nul
    if not errorlevel 1 set "PYTHON_CMD=py -3.11"
)

if not defined PYTHON_CMD (
    py -3 --version >nul 2>nul
    if not errorlevel 1 set "PYTHON_CMD=py -3"
)

if not defined PYTHON_CMD (
    python --version >nul 2>nul
    if not errorlevel 1 set "PYTHON_CMD=python"
)

if not defined PYTHON_CMD (
    echo [ERROR] Python was not found.
    echo Install Python 3.11 or 3.12 from python.org, then run this file again.
    echo.
    pause
    exit /b 1
)

echo Using Python:
%PYTHON_CMD% --version
echo.

REM ------------------------------------------------------------
REM Create requirements.txt if it is missing.
REM ------------------------------------------------------------
if not exist "requirements.txt" (
    echo Creating requirements.txt ...
    > requirements.txt echo numpy
    >> requirements.txt echo scipy
    >> requirements.txt echo matplotlib
    >> requirements.txt echo numba
    >> requirements.txt echo ase
    >> requirements.txt echo python-docx
)

REM ------------------------------------------------------------
REM Create virtual environment.
REM ------------------------------------------------------------
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment .venv ...
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 (
        echo.
        echo [ERROR] Failed to create .venv.
        echo Try installing Python 3.12 and run this file again.
        echo.
        pause
        exit /b 1
    )
) else (
    echo Virtual environment already exists: .venv
)

echo.
echo Activating .venv ...
call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Could not activate .venv
    pause
    exit /b 1
)

echo.
echo Upgrading pip/setuptools/wheel ...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo.
    echo [ERROR] pip upgrade failed.
    echo Check your internet connection or proxy/firewall.
    echo.
    pause
    exit /b 1
)

echo.
echo Installing project libraries from requirements.txt ...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Library installation failed.
    echo If this happened on Python 3.13, install Python 3.12 and run this file again.
    echo.
    pause
    exit /b 1
)

echo.
echo Checking imports ...
python -c "import numpy, scipy, matplotlib, ase, numba; import docx; print('OK: all Python libraries imported successfully')"
if errorlevel 1 (
    echo.
    echo [ERROR] Import check failed.
    echo.
    pause
    exit /b 1
)

REM ------------------------------------------------------------
REM Create simple launcher for main.py.
REM ------------------------------------------------------------
echo Creating RUN_main.bat ...
> RUN_main.bat echo @echo off
>> RUN_main.bat echo cd /d "%%~dp0"
>> RUN_main.bat echo call ".venv\Scripts\activate.bat"
>> RUN_main.bat echo python main.py
>> RUN_main.bat echo echo.
>> RUN_main.bat echo echo Finished. Press any key to close.
>> RUN_main.bat echo pause ^>nul

echo.
echo ============================================================
echo  Setup finished successfully.
echo ============================================================
echo.
echo You can now run the project by double-clicking:
echo   RUN_main.bat
echo.
echo Important:
echo   MOPAC itself is external. Check settings\config.py:
echo   MOPAC_EXE_PATH must point to your real MOPAC executable.
echo.
pause
