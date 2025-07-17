#!/bin/bash

set -e  # Exit on error

echo "This project only works with Python versions 3.12.7 and below."
echo "(Optional) Create a virtual environment with Python 3.12 and install required libraries."
echo "Type y/yes to proceed, or anything else to only install libraries system-wide."
read -p "Would you like to create a virtual environment? (y/yes): " create_venv

# Normalize input to lowercase and check if starts with 'y'
if [[ "${create_venv,,}" =~ ^y ]]; then
    echo "Creating virtual environment 'palenv' with Python 3.12..."

    if ! command -v python3.12 &>/dev/null; then
        echo "Python 3.12 not found. Please install Python 3.12 before proceeding."
        exit 1
    fi

    python3.12 -m venv palenv
    source palenv/bin/activate

    echo "Virtual environment 'palenv' activated."

    if [[ -f "requirements.txt" ]]; then
        echo "Installing dependencies from requirements.txt..."
        pip install --upgrade pip
        pip install -r requirements.txt
    else
        echo "requirements.txt not found. Skipping package installation."
    fi
    
    # Move src into palenv
    if [[ -d "src" ]]; then
        echo "Moving 'src' folder into palenv..."
        mv src palenv/
    else
        echo "'src' folder not found. Skipping move."
    fi

    # Move data into palenv
    if [[ -d "data" ]]; then
        echo "Moving 'data' folder into palenv..."
        mv data palenv/
    else
        echo "'data' folder not found. Skipping move."
    fi

    deactivate
    echo "Virtual environment deactivated."
else
    echo "Skipping virtual environment setup."
    echo "Installing dependencies system-wide..."

    if [[ -f "requirements.txt" ]]; then
        pip3 install --upgrade pip
        pip3 install -r requirements.txt
    else
        echo "requirements.txt not found. Skipping package installation."
    fi
fi

echo "Cleaning up setup files..."
rm -f setup.sh
rm -f startup.bat
rm -f requirements.txt

echo "Setup complete."