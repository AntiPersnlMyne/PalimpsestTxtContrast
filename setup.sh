#!/bin/bash

# Exit immediately if any command fails
set -e

# Prompt for virtual environment creation
echo "This project only works with Python versions 3.12.7 and below."
read -p "Would you like to create a virtual environment (palenv)? (y/yes): " create_venv

# Function to create directory structure
create_directory_structure() {
    echo "Creating directory structure..."
    mkdir -p data/input
    mkdir -p data/output
    mkdir -p src
    mkdir -p src/python_scripts
    mkdir -p src/IDL_scripts
}

# Function to install dependencies
install_dependencies() {
    if [ -f "requirements.txt" ]; then
        echo "Installing dependencies from requirements.txt..."
        pip install --upgrade pip
        pip install -r requirements.txt
    else
        echo "requirements.txt not found. Skipping package installation."
    fi
}

# Create directory structure regardless of user input
create_directory_structure

# Check if user wants to create virtual environment
if [[ "$create_venv" == "y" || "$create_venv" == "yes" ]]; then
    # Create virtual environment with Python 3.12.7
    python3.12 -m venv palenv
    
    # Activate virtual environment
    source palenv/bin/activate
    
    # Install dependencies in virtual environment
    install_dependencies
    
    # Move src file(s) to directory
    mv main.py src/
    
    # Deactivate virtual environment
    deactivate
else
    # Install dependencies system-wide
    install_dependencies
    
    # Move src file(s) to directory
    mv main.py src/
fi

# Report setup complete
echo "Setup complete."

# Delete the setup files
rm "$0"
rm "setup.bat"
