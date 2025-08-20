#!/usr/bin/env bash
set -e

echo "Checking Python version..."
PYTHON_VERSION=$(python3 -V 2>&1 | awk '{print $2}')
REQUIRED="3.13"
if [[ "$(printf '%s\n' "$REQUIRED" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED" ]]; then
  echo "Python 3.13+ is required, found $PYTHON_VERSION"
  exit 1
fi

echo "Do you want to create a virtual environment (gospenv)? [y/n]"
read create_venv

if [[ "$create_venv" == "y" || "$create_venv" == "Y" ]]; then
  echo "Creating virtual environment 'gospenv' ..."
  python3 -m venv gospenv
  source gospenv/bin/activate
fi

echo "Installing Python requirements ..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo "Building Cython files ..."
pip install -e .

echo "Cleaning up setup files ..."
rm -f setup.sh setup.bat requirements.txt

if [[ "$create_venv" == "y" || "$create_venv" == "Y" ]]; then
    echo "[Hint] Move `gosp/`, `data/`, and `main.py` into gospenv root.

echo "Setup complete"
