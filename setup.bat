@echo off
setlocal enabledelayedexpansion

:: Prompt user for virtual environment creation
echo (Optional) Create a virtual environment with compatible python version and necessary libraries.
echo Type y/yes to proceed, or n/no to only install libraries.
set /p create_venv="Would you like to create a virtual environment? (y/yes): "

:: Check if input starts with 'y' or 'Y'
if /i "%create_venv:~0,1%"=="y" (
    echo Creating virtual environment '.venv' with Python 3.12...

    :: Ensure Python 3.12 is installed and used
    py -3.12 -m venv venv
    if errorlevel 1 (
        echo Failed to create virtual environment with Python 3.12.
        goto InstallOnly
    )

    call venv\Scripts\activate.bat

    echo Virtual environment 'venv' activated.

    :: Upgrade pip and install dependencies
    if exist "requirements.txt" (
        echo Installing dependencies from requirements.txt...
        pip install --upgrade pip
        pip install -r requirements.txt
    ) else (
        echo requirements.txt not found.
    )

    :: Move src folder into venv
    if exist "src" (
        echo Moving 'src' folder into venv...
        move /Y src venv\
    ) else (
        echo 'src' folder not found. Skipping move.
    )

    :: Move data folder into venv
    if exist "data" (
        echo Moving 'data' folder into venv...
        move /Y data venv\
    ) else (
        echo 'data' folder not found. Skipping move.
    )

    :: Deactivate virtual environment
    echo Deactivating virtual environment...
    call venv\Scripts\deactivate.bat

    goto Cleanup
)

:InstallOnly
echo Skipping virtual environment setup.
echo Installing dependencies system-wide...

if exist "requirements.txt" (
    pip install --upgrade pip
    pip install -r requirements.txt
) else (
    echo requirements.txt not found. Skipping package installation.
)

:Cleanup
echo Cleaning up setup files...
del "%~f0" >nul 2>&1
del "startup.sh" >nul 2>&1
del "requirements.txt" >nul 2>&1
echo Setup complete.
