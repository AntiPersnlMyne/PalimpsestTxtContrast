@echo off

rem Exit immediately if any command fails
if errorlevel 1 exit /b 1

rem Install required packages
if exist requirements.txt (
    echo Installing dependencies from requirements.txt...
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) else (
    echo requirements.txt not found. Skipping package installation.
)

rem Create the directory structure
echo Creating directory structure...
if not exist data\input mkdir data\input
if not exist data\output mkdir data\output
if not exist src mkdir src
if not exist src\python_scripts mkdir src\python_scripts
if not exist src\IDL_scripts mkdir src\IDL_scripts

rem Move src file(s) to directory
move main.py envi\src

rem Report setup complete
echo Setup complete.

rem Delete the setup.bat file
del setup.sh
del setup.bat
