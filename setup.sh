#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define the name for the virtual environment
VENV_NAME="bert-env"

echo "Creating virtual environment: $VENV_NAME..."
# Create a virtual environment using python3's venv module
python3 -m venv $VENV_NAME

echo "Activating virtual environment..."
# Activate the virtual environment
source $VENV_NAME/bin/activate

echo "Installing dependencies from requirements.txt..."
# Upgrade pip to the latest version and install the required packages
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete. The '$VENV_NAME' environment is ready and activated."
echo "To deactivate, simply run: deactivate"
