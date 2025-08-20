@echo off
setlocal enabledelayedexpansion

echo Checking Python version...
for /f "tokens=2 delims= " %%a in ('python -V 2^>^&1') do set PYTHON_VERSION=%%a
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set MAJOR=%%a
    set MINOR=%%b
)

if %MAJOR% lss 3 (
    echo Python 3.13+ required, found %PYTHON_VERSION%
    exit /b 1
)
if %MAJOR%==3 if %MINOR% lss 13 (
    echo Python 3.13+ required, found %PYTHON_VERSION%
    exit /b 1
)

set /p CREATE_VENV="Do you want to create a virtual environment (gospenv)? [y/n]: "
if /i "%CREATE_VENV%"=="y" (
    echo Creating virtual environment 'gospenv' ...
    python -m venv gospenv
    call gospenv\Scripts\activate
)

echo Installing Python requirements ...
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo Building Cython files ...
pip install -e .

echo Cleaning up setup files ...
del setup.sh
del requirements.txt
del setup.bat

echo Setup complete!