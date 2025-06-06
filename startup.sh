#!/bin/bash

# Exit immediately if any command fails
set -e

# Create a Python virtual environment named "envi"
echo "Creating virtual environment 'envi'..."
python3 -m venv envi

# Activate the virtual environment
echo "Activating virtual environment..."
source envi/bin/activate

# Install required packages
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "requirements.txt not found. Skipping package installation."
fi

# Create the directory structure
echo "Creating directory structure..."
mkdir -p envi/data/input
mkdir -p envi/data/output
mkdir -p envi/src

echo "Setup complete."
