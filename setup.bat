@echo off
setlocal enabledelayedexpansion

:: Exit immediately if any command fails
set "ERRORLEVEL=0"

:: Prompt for virtual environment creation
echo This project only works with Python versions 3.12.7 and below.
set /p create_venv="Would you like to create a virtual environment (palenv), than includes a working Python version? (y/yes): "

:: Function to create directory structure
:CreateDirectoryStructure
echo Creating directory structure...
mkdir data\input
mkdir data\output
mkdir src
mkdir src\python_scripts
mkdir src\IDL_scripts
goto :eof

:: Function to install dependencies
:InstallDependencies
if exist "requirements.txt" (
    echo Installing dependencies from requirements.txt...
    pip install --upgrade pip
    pip install -r requirements.txt
) else (
    echo requirements.txt not found. Skipping package installation.
)
goto :eof

:: Create directory structure regardless of user input
call :CreateDirectoryStructure

:: Check if user wants to create virtual environment
if /i "%create_venv%"=="y" (
    if /i "%create_venv%"=="yes" (
        :: Create virtual environment with Python 3.12.7
        python -m venv palenv
        
        :: Activate virtual environment
        call palenv\Scripts\activate.bat
        
        :: Install dependencies in virtual environment
        call :InstallDependencies
        
        :: Move src file(s) to directory
        move main.py src\
        
        :: Deactivate virtual environment
        deactivate
    )
) else (
    :: Install dependencies system-wide
    call :InstallDependencies
    
    :: Move src file(s) to directory
    move main.py src\
)

:: Report setup complete
echo Setup complete.

:: Delete the setup.bat file
del "%~f0"
