#!/bin/bash
set -e

echo "Checking Python version ..."
PYTHON_VERSION=$(python3 -V 2>&1 | awk '{print $2}')
MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$MAJOR" -lt 3 ] || { [ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 13 ]; }; then
    echo "Python 3.13+ required, found $PYTHON_VERSION"
    exit 1
fi

read -p "Do you want to create a virtual environment (gospenv)? [y/n]: " CREATE_VENV
if [[ "$CREATE_VENV" =~ ^[Yy]$ ]]; then
    echo "Creating virtual environment 'gospenv' ..."
    python3 -m venv gospenv
    source gospenv/bin/activate
fi

echo "Installing Python requirements ..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo "Building Cython files..."
pip install -e .

echo "Moving compiled Cython files to gosp/build ..."
mkdir -p gosp/build
# Find all .so files (Linux equivalent of .pyd) in build/lib/gosp
for f in $(find build/lib/gosp -name "*.so"); do
    echo "Moving $f to gosp/build/"
    mv "$f" gosp/build/
done

echo "Cleaning up setup files ..."
rm -f requirements.txt pyproject.toml setup.sh

echo "Setup complete!"
