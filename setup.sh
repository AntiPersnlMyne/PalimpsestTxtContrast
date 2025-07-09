#!/bin/bash

# Exit immediately if any command fails
set -e

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
mkdir -p data/input
mkdir -p data/output
mkdir -p src
mkdir -p src/python_scripts
mkdir -p src/IDL_scripts

# Move src file(s) to directory
mv main.py envi/src

# Report setup complete
echo "Setup complete."

# Delete the setup.sh file
rm setup.bat
rm "$0"
