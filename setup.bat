@echo off

rem Exit immediately if any command fails
if errorlevel 1 exit /b 1

rem Create a Python virtual environment named "envi"
echo Creating virtual environment 'envi'...
python -m venv envi

rem Activate the virtual environment
echo Activating virtual environment...
call envi\Scripts\activate.bat

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
if not exist envi\data\input mkdir envi\data\input
if not exist envi\data\output mkdir envi\data\output
if not exist envi\src mkdir envi\src

rem Move src file(s) to directory
move ENVI_script.py envi\src

rem Report setup complete
echo Setup complete.

rem Delete the setup.bat file
del setup.bat
del setup.sh
