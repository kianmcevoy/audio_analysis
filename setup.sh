#!/usr/bin/env bash
set -e

# ----------------------------------------
# Local Python environment setup
# ----------------------------------------

# Choose python executable (override by exporting PYTHON=python3.x)
PYTHON=${PYTHON:-python3}

VENV_DIR=".venv"
REQ_FILE="requirements.txt"

echo "Using Python: $($PYTHON --version)"

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR"
    $PYTHON -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists: $VENV_DIR"
fi

# Activate venv
echo "Activating virtual environment"
source "$VENV_DIR/bin/activate"

# Upgrade pip (local to venv)
echo "Upgrading pip"
pip install --upgrade pip

# Install requirements
if [ -f "$REQ_FILE" ]; then
    echo "Installing dependencies from $REQ_FILE"
    pip install -r "$REQ_FILE"
else
    echo "ERROR: $REQ_FILE not found"
    exit 1
fi

echo
echo "----------------------------------------"
echo "Environment setup complete"
echo
echo "To use:"
echo "  source .venv/bin/activate"
echo
echo "To run analysis:"
echo "  python -m analyse.cli --help"
echo "----------------------------------------"
